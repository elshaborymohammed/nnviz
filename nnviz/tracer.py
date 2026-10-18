"""Trace a raw ``nn.Module`` into a fine-grained dataflow ``Graph``.

Tracing runs one real forward pass with:
  * forward hooks on every module (a node per module call), and
  * a ``TorchFunctionMode`` that sees functional ops such as ``x + y``,
    ``torch.cat`` or ``F.softmax`` — this is what makes residual / skip
    connections visible, since they are usually plain additions.

Tensors are followed by object identity, so models with Python control flow,
HuggingFace models, and anything else that runs eagerly can be traced.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from .graph import Graph, Node

# Ops that only rearrange / view data: the output inherits the producer of the
# input instead of becoming a node of its own.
PASS_THROUGH = {
    "view", "reshape", "permute", "transpose", "contiguous", "unsqueeze",
    "squeeze", "expand", "expand_as", "flatten", "unflatten", "__getitem__",
    "to", "float", "half", "bfloat16", "double", "type", "type_as", "clone",
    "detach", "repeat", "repeat_interleave", "narrow", "split", "chunk",
    "unbind", "t", "movedim", "masked_fill", "masked_fill_", "cpu", "cuda",
    "requires_grad_",
    "__setitem__", "view_as", "reshape_as", "select", "index_select",
}
# Functional dropout calls; drawn as Dropout blocks (``hide_dropout`` removes them).
DROPOUT_OPS = {"dropout", "alpha_dropout", "feature_alpha_dropout",
               "dropout1d", "dropout2d", "dropout3d"}
# Arithmetic ops that are hidden when only one operand is a tensor (scaling
# by a constant, e.g. ``emb(x) * sqrt(d_model)``).
SCALAR_ARITH = {
    "add", "__add__", "__radd__", "add_", "__iadd__", "sub", "__sub__",
    "__rsub__", "sub_", "__isub__", "mul", "__mul__", "__rmul__", "mul_",
    "__imul__", "div", "__truediv__", "__rtruediv__", "div_", "__itruediv__",
    "pow", "__pow__", "neg", "__neg__",
}
# Ops never worth showing (they produce no tensor data flow of interest).
IGNORED = {
    "size", "dim", "numel", "__len__", "__repr__", "__format__", "item",
    "tolist", "__bool__", "__iter__", "__contains__", "is_floating_point",
    "is_contiguous", "__get__", "__set__", "data_ptr", "element_size",
    "__hash__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "eq", "ne", "lt", "le", "gt", "ge", "all", "any", "isnan", "isinf",
    "__and__", "__or__", "__invert__", "logical_not", "logical_and",
    "logical_or", "triu", "tril", "ones_like", "zeros_like", "full_like",
    "empty_like", "new_ones", "new_zeros", "new_full", "new_empty",
    "new_tensor", "arange", "zeros", "ones", "full", "empty", "rand",
    "randn", "tensor", "as_tensor", "finfo", "iinfo", "cumsum",
    "__index__", "get_device", "__reduce_ex__", "__deepcopy__",
}

_OP_ALIASES = {
    "__add__": "add", "__radd__": "add", "add_": "add", "__iadd__": "add",
    "__sub__": "sub", "__rsub__": "sub", "sub_": "sub", "__isub__": "sub",
    "__mul__": "mul", "__rmul__": "mul", "mul_": "mul", "__imul__": "mul",
    "__truediv__": "div", "__rtruediv__": "div", "div_": "div",
    "__itruediv__": "div", "__matmul__": "matmul", "__rmatmul__": "matmul",
    "bmm": "matmul", "mm": "matmul", "concat": "cat", "concatenate": "cat",
}


def _op_name(func) -> str:
    name = getattr(func, "__name__", None) or str(func)
    if name in ("__get__", "__set__"):
        # property access such as Tensor.shape / Tensor.T
        owner = getattr(func, "__self__", None)
        name = getattr(owner, "__name__", name)
    return name


def flatten_tensors(obj: Any, out: Optional[list] = None, _seen=None) -> List[torch.Tensor]:
    """Collect tensors from arbitrarily nested containers / dataclasses."""
    if out is None:
        out, _seen = [], set()
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple, set)):
        for o in obj:
            flatten_tensors(o, out, _seen)
    elif isinstance(obj, dict):
        for o in obj.values():
            flatten_tensors(o, out, _seen)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if id(obj) not in _seen:
            _seen.add(id(obj))
            for f in dataclasses.fields(obj):
                flatten_tensors(getattr(obj, f.name, None), out, _seen)
    return out


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class Tracer:
    """Run a forward pass and record the dataflow graph.

    Args:
        model: any ``nn.Module``.
        ignore_ops: extra op names to treat as pass-through.
        leaf_types: extra module classes to record as single nodes.
    """

    def __init__(self, model: nn.Module, ignore_ops=(), leaf_types=()):
        self.model = model
        self.pass_through = PASS_THROUGH | set(ignore_ops)
        # Modules drawn as a single node even though they have children.
        self.leaf_types = (nn.MultiheadAttention,) + tuple(leaf_types)

    # ----------------------------------------------------------------- public
    def trace(self, inputs: Any) -> Graph:
        self.graph = Graph()
        self._producer: Dict[int, str] = {}   # id(tensor) -> node id
        self._shape: Dict[int, tuple] = {}
        self._keepalive: list = []            # stop id() reuse during trace
        self._stack: List[str] = []           # module qualnames being executed
        self._leaf_depth = 0                  # >0 while inside a leaf module
        self._counter = 0
        self._calls: Dict[str, int] = {}
        self._names = {m: n for n, m in self.model.named_modules()}
        self._param_names = {id(p): n for n, p in self.model.named_parameters()}
        self._param_names.update({id(b): n for n, b in self.model.named_buffers()})

        args, kwargs = self._normalise_inputs(inputs)
        self._register_inputs(args, kwargs)

        handles = []
        for module in self.model.modules():
            handles.append(module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(self._post_hook, with_kwargs=True))
        was_training = self.model.training
        self.model.eval()
        try:
            # Gradients stay enabled: PyTorch's Transformer layers take a fused
            # "fast path" (bypassing their sub-modules) under no_grad + eval.
            with _Recorder(self):
                output = self.model(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            self.model.train(was_training)

        self._register_outputs(output)
        self.graph.meta["total_params"] = count_params(self.model)
        self.graph.meta["trainable_params"] = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        self.graph.meta["model_class"] = type(self.model).__name__
        return self.graph

    # ------------------------------------------------------------ bookkeeping
    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _set_producer(self, tensor: torch.Tensor, nid: str) -> None:
        self._producer[id(tensor)] = nid
        self._shape[id(tensor)] = tuple(tensor.shape)
        self._keepalive.append(tensor)

    def _producer_of(self, tensor: torch.Tensor) -> Optional[str]:
        nid = self._producer.get(id(tensor))
        if nid is None and id(tensor) in self._param_names:
            nid = self._param_node(tensor)
        return nid

    def _param_node(self, tensor: torch.Tensor) -> str:
        """A parameter/buffer used directly in a functional op (e.g. a learned
        positional embedding or a CLS token) becomes a small source node."""
        name = self._param_names[id(tensor)]
        nid = f"param_{name}"
        if nid not in self.graph.nodes:
            owner = name.rsplit(".", 1)[0] if "." in name else ""
            self.graph.add_node(Node(
                nid, "param", "Parameter", name=name, path=owner,
                params=tensor.numel() if isinstance(tensor, nn.Parameter) else 0,
                order=self._counter, meta={"shape": tuple(tensor.shape)}))
            self._counter += 1
        self._set_producer(tensor, nid)
        return nid

    def _connect_inputs(self, nid: str, tensors) -> List[str]:
        srcs = []
        for t in tensors:
            src = self._producer_of(t)
            if src is not None:
                self.graph.add_edge(src, nid, tuple(t.shape))
                if src not in srcs:
                    srcs.append(src)
        return srcs

    def _normalise_inputs(self, inputs):
        if isinstance(inputs, dict):
            return (), dict(inputs)
        if isinstance(inputs, (tuple, list)):
            return tuple(inputs), {}
        return (inputs,), {}

    def _register_inputs(self, args, kwargs):
        try:
            sig = inspect.signature(self.model.forward)
            bound = sig.bind_partial(*args, **kwargs).arguments
        except (TypeError, ValueError):
            bound = {f"input_{i}": a for i, a in enumerate(args)}
            bound.update(kwargs)
        for name, value in bound.items():
            tensors = flatten_tensors(value)
            if not tensors:
                continue
            nid = f"input_{name}"
            self.graph.add_node(Node(nid, "input", "Input", name=name,
                                     order=self._counter,
                                     meta={"shape": tuple(tensors[0].shape)}))
            self._counter += 1
            for t in tensors:
                self._set_producer(t, nid)

    def _register_outputs(self, output):
        picked = _pick_outputs(output)
        for name, tensor in picked:
            src = self._producer_of(tensor)
            if src is None:
                continue
            nid = f"output_{name}"
            self.graph.add_node(Node(nid, "output", "Output", name=name,
                                     order=self._counter + 1,
                                     meta={"shape": tuple(tensor.shape)}))
            self._counter += 1
            self.graph.add_edge(src, nid, tuple(tensor.shape))
        self.graph.prune_to_outputs()

    # ------------------------------------------------------------------ hooks
    def _is_leaf(self, module) -> bool:
        return isinstance(module, self.leaf_types) or not any(True for _ in module.children())

    def _is_root_scope(self, module) -> bool:
        # The root is only a scope, unless the whole model is a single layer.
        return module is self.model and not self._is_leaf(module)

    def _pre_hook(self, module, args, kwargs):
        name = self._names.get(module, type(module).__name__)
        self._stack.append(name)
        if self._leaf_depth or self._is_leaf(module):
            self._leaf_depth += 1
        if self._leaf_depth > 1 or self._is_root_scope(module):
            return None
        # Remember what flowed into this call; the node is created on exit so
        # that execution order of ops inside it is preserved.
        module._nnviz_in = flatten_tensors(args) + flatten_tensors(kwargs)
        module._nnviz_kwargs = {
            k: v for k, v in kwargs.items()
            if isinstance(v, (bool, int, float, str)) or v is None
            or isinstance(v, torch.Tensor)}
        module._nnviz_nargs = len(args)
        return None

    def _post_hook(self, module, args, kwargs, output):
        name = self._stack.pop()
        leaf_depth = self._leaf_depth
        if self._leaf_depth:
            self._leaf_depth -= 1
        if leaf_depth > 1 or self._is_root_scope(module):
            return None
        is_leaf = leaf_depth == 1
        call = self._calls.get(name, 0)
        self._calls[name] = call + 1
        nid = f"mod_{name or 'root'}" + (f"#{call}" if call else "")
        parent = self._stack[-1] if self._stack else ""
        node = Node(nid, "module", type(module).__name__, name=name,
                    path=name, order=self._counter,
                    params=count_params(module),
                    meta={"leaf": is_leaf, "parent": parent, "call": call,
                          "module": module,
                          "kwargs": getattr(module, "_nnviz_kwargs", {})})
        self._counter += 1
        ins, outs = getattr(module, "_nnviz_in", []), flatten_tensors(output)
        node.meta["in_shape"] = tuple(ins[0].shape) if ins else None
        node.meta["out_shape"] = tuple(outs[0].shape) if outs else None
        if is_leaf:
            # Leaf modules are drawn as one node: wire their inputs/outputs.
            self.graph.add_node(node)
            self._connect_inputs(nid, getattr(module, "_nnviz_in", []))
            for t in flatten_tensors(output):
                self._set_producer(t, nid)
        else:
            # Container modules are only scope markers for their contents.
            self.graph.meta.setdefault("containers", {})[name] = node
        for attr in ("_nnviz_in", "_nnviz_kwargs", "_nnviz_nargs"):
            if hasattr(module, attr):
                delattr(module, attr)
        return None

    def _on_function(self, func, args, kwargs, output):
        if self._leaf_depth:
            return
        name = _op_name(func)
        if name in IGNORED or (name.startswith("_") and name not in _OP_ALIASES
                               and name not in SCALAR_ARITH and name != "__getitem__"):
            return
        in_tensors = flatten_tensors(args) + flatten_tensors(kwargs)
        out_tensors = flatten_tensors(output)
        if not out_tensors:
            return
        known = [t for t in in_tensors if self._producer_of(t) is not None]
        if not known:
            return  # constants / masks built from scratch
        canonical = _OP_ALIASES.get(name, name)
        if name in self.pass_through or (name in SCALAR_ARITH and len(known) == 1) \
                or (canonical in ("add", "sub", "mul", "div") and len(known) == 1):
            src = self._producer_of(known[0])
            for t in out_tensors:
                self._set_producer(t, src)
            return
        nid = self._new_id(f"op_{canonical}")
        scope = self._stack[-1] if self._stack else ""
        node = self.graph.add_node(Node(nid, "op", canonical, name=canonical,
                                        path=scope, order=self._counter))
        self._connect_inputs(nid, known)
        if kwargs.get("is_causal") is True:  # attn_mask may only be padding
            node.meta["masked"] = True
        if canonical in DROPOUT_OPS:
            p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
            node.detail = f"p={p:g}"
        for t in out_tensors:
            self._set_producer(t, nid)


class _Recorder(TorchFunctionMode):
    def __init__(self, tracer: Tracer):
        super().__init__()
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self.tracer._on_function(func, args, kwargs, out)
        except Exception:  # never let visualisation break the forward pass
            pass
        return out


_PREFERRED_OUTPUTS = ("logits", "prediction_logits", "last_hidden_state",
                      "pooler_output", "out", "output")


def _pick_outputs(output) -> List[tuple]:
    """Choose which returned tensors count as model outputs."""
    if isinstance(output, torch.Tensor):
        return [("output", output)]
    if isinstance(output, dict) or (dataclasses.is_dataclass(output)
                                    and not isinstance(output, type)):
        items = dict(output) if isinstance(output, dict) else {
            f.name: getattr(output, f.name) for f in dataclasses.fields(output)}
        for key in _PREFERRED_OUTPUTS:
            if isinstance(items.get(key), torch.Tensor):
                return [(key, items[key])]
        for key, val in items.items():
            if isinstance(val, torch.Tensor):
                return [(key, val)]
    if isinstance(output, (tuple, list)):
        tensors = [o for o in output if isinstance(o, torch.Tensor)]
        if tensors:
            return [("output", tensors[0])]
    tensors = flatten_tensors(output)
    return [("output", tensors[0])] if tensors else []
