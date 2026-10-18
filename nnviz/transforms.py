"""Graph rewriting passes.

Each pass is a plain function ``pass(graph, ...) -> None`` that mutates the
graph, so pipelines can be reordered, extended or replaced freely.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional

from torch import nn

from .analysis import adjacency, find_residuals, is_merge_add
from .graph import Graph, Node
from .style import categorize, friendly_name, module_detail
from .tracer import count_params


def _in_scope(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + ".")


def _common_scope(paths: List[str]) -> str:
    parts = [p.split(".") for p in paths]
    common = []
    for level in zip(*parts):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    return ".".join(common)


# ------------------------------------------------------------------- cleanup
def drop_nodes(graph: Graph, types: Iterable[str] = ("Dropout", "Identity"),
               categories: Iterable[str] = ("dropout",)) -> None:
    """Dissolve no-op layers such as Dropout / Identity."""
    types, categories = set(types), set(categories)
    for n in list(graph.nodes.values()):
        if n.kind in ("module", "op") and (n.type in types or categorize(n) in categories):
            graph.dissolve(n.id)


def hide_inputs(graph: Graph, pattern: Optional[str] = r"mask") -> None:
    """Remove auxiliary inputs (attention masks, ...) matching ``pattern``."""
    if not pattern:
        return
    for n in list(graph.nodes.values()):
        if n.kind == "input" and re.search(pattern, n.name, re.IGNORECASE):
            graph.remove_node(n.id)


def record_attention_flags(graph: Graph) -> None:
    """Flag attention calls that received a causal mask (from call kwargs)."""
    for n in graph.nodes.values():
        kw = n.meta.get("kwargs") or {}
        if kw.get("is_causal") is True or kw.get("attn_mask") is not None \
                or kw.get("tgt_mask") is not None:
            n.meta["masked"] = True


# ------------------------------------------------------------- collapsing
def default_collapse_rule(pattern: str = r"attention|attn|mlp|feed_?forward|ffn|intermediate"
                                         r"|embedding|positional|pos_?enc|rotary"):
    regex = re.compile(pattern, re.IGNORECASE)
    return lambda name, module: bool(regex.search(type(module).__name__))


def collapse_modules(graph: Graph, model: nn.Module,
                     rule: Callable[[str, nn.Module], bool],
                     max_depth: Optional[int] = None,
                     keep_residuals: bool = True) -> None:
    """Merge every node inside a module scope into one node.

    A container is collapsed when ``rule(name, module)`` is true or when its
    depth equals ``max_depth``.  With ``keep_residuals`` a module that contains
    a skip connection is left expanded, so the connection stays visible.
    """
    modules = dict(model.named_modules())
    merge_points = {r.merge for r in find_residuals(graph, mark_edges=False)}
    scopes = sorted({n.path for n in graph.nodes.values() if n.kind == "op"}
                    | set(graph.meta.get("containers", {})),
                    key=lambda s: (s.count("."), s))
    for scope in scopes:
        if not scope or scope not in modules:
            continue
        module = modules[scope]
        depth = scope.count(".") + 1
        if not (rule(scope, module) or (max_depth is not None and depth >= max_depth)):
            continue
        members = [n for n in graph.nodes.values()
                   if _in_scope(n.path, scope) and n.kind in ("module", "op", "param")]
        if not members:
            continue
        ids = {n.id for n in members}
        if keep_residuals and not (max_depth is not None and depth >= max_depth) \
                and merge_points & ids:
            continue  # collapsing would hide a skip connection
        ops = {n.type for n in members if n.kind == "op"}
        container = graph.meta.get("containers", {}).get(scope)
        merged = Node(
            id=f"mod_{scope}", kind="module", type=type(module).__name__, name=scope,
            path=scope, params=count_params(module),
            order=max(n.order for n in members),
            meta={"module": module, "collapsed": True, "inner_ops": ops,
                  "masked": any(n.meta.get("masked") for n in members),
                  "kwargs": container.meta.get("kwargs", {}) if container else {},
                  "in_shape": container.meta.get("in_shape") if container else None,
                  "out_shape": container.meta.get("out_shape") if container else None})
        graph.merge(ids, merged)


def _signature(module: nn.Module) -> tuple:
    # Class + direct children; the node-sequence check in collapse_repeats
    # catches real structural differences (e.g. T5's extra bias in block 0).
    return (type(module).__name__, tuple(n for n, _ in module.named_children()))


def collapse_repeats(graph: Graph, model: nn.Module, min_repeat: int = 2) -> None:
    """Fold runs of identical sibling layers (e.g. ``layers.0 … layers.11``)
    into the first instance, remembered as an ``N×`` group."""
    groups = graph.meta.setdefault("repeats", {})
    containers = sorted(((n, m) for n, m in model.named_modules()),
                        key=lambda nm: nm[0].count(".") if nm[0] else -1)
    for cname, container in containers:
        # Only indexed stacks (ModuleList / Sequential: layers.0, layers.1, ...)
        children = [(c, m) for c, m in container.named_children() if c.isdigit()]
        i = 0
        while i < len(children):
            j = i + 1
            while j < len(children) and _signature(children[j][1]) == _signature(children[i][1]):
                j += 1
            run = [f"{cname}.{c}" if cname else c for c, _ in children[i:j]]
            i = j
            if len(run) < min_repeat:
                continue
            instances = [sorted((n for n in graph.nodes.values()
                                 if n.kind != "input" and _in_scope(n.path, r)),
                                key=lambda n: n.order) for r in run]
            instances = [inst for inst in instances if inst]
            if len(instances) < min_repeat:
                continue
            base = instances[0]
            if any([n.type for n in inst] != [n.type for n in base] for inst in instances):
                continue
            mapping: Dict[str, str] = {}
            which: Dict[str, int] = {}
            for k, inst in enumerate(instances):
                for n, b in zip(inst, base):
                    mapping[n.id] = b.id
                    which[n.id] = k
            old_edges, graph.edges = graph.edges, []
            for e in old_edges:
                ks, kd = which.get(e.src), which.get(e.dst)
                if ks is not None and kd is not None and ks != kd:
                    continue  # chaining between instances becomes implicit
                graph.add_edge(mapping.get(e.src, e.src), mapping.get(e.dst, e.dst), e.shape, e.kind)
            for inst in instances[1:]:
                for n in inst:
                    graph.nodes.pop(n.id, None)
            for n in base:
                n.meta["repeat"] = run[0]
            first = model.get_submodule(run[0])
            groups[run[0]] = {"count": len(instances), "scope": run[0],
                              "type": type(first).__name__, "params": count_params(first)}


# ----------------------------------------------------------------- merging
FFN_CATEGORIES = {"linear", "activation", "dropout", "ffn", "merge"}


def merge_feed_forward(graph: Graph) -> None:
    """Collapse the body of a residual branch made only of linear layers and
    activations (``Linear → ReLU → Linear``, SwiGLU, ...) into ``Feed Forward``."""
    for res in find_residuals(graph, mark_edges=False):
        body = [graph.nodes[n] for n in res.branch if n in graph.nodes]
        body = [n for n in body if categorize(n) != "norm"]
        # Dropout on the branch output belongs to the sublayer (FFN → Dropout →
        # Add & Norm), so keep it as its own block; inner dropout stays merged.
        ids = {n.id for n in body}
        while True:
            exits = {e.src for e in graph.edges if e.src in ids and e.dst not in ids}
            tail = {i for i in exits if categorize(graph.nodes[i]) == "dropout"}
            if not tail:
                break
            ids -= tail
        body = [n for n in body if n.id in ids]
        if len(body) < 2 or not all(categorize(n) in FFN_CATEGORIES for n in body):
            continue
        if not any(categorize(n) in ("linear", "ffn") for n in body):
            continue
        ids = {n.id for n in body}
        entries = {e.src for e in graph.edges if e.dst in ids and e.src not in ids}
        if len(entries) != 1:
            continue
        detail = _width_detail(graph, ids)
        scope = _common_scope([n.path for n in body])
        graph.merge(ids, Node(
            id=f"ffn_{min(ids)}", kind="module", type="FeedForward",
            name=scope, path=scope,
            params=sum(n.params for n in body), order=max(n.order for n in body),
            label="Feed Forward", detail=detail, category="ffn",
            meta={"repeat": body[0].meta.get("repeat")}))


def merge_add_norm(graph: Graph) -> None:
    """Fuse ``add → LayerNorm`` (post-norm residual) into one ``Add & Norm`` box."""
    preds, succs = adjacency(graph)
    for nid in list(graph.nodes):
        if nid not in graph.nodes or not is_merge_add(graph, nid):
            continue
        nxt = succs.get(nid, [])
        if len(nxt) != 1 or nxt[0] not in graph.nodes:
            continue
        norm = graph.nodes[nxt[0]]
        if categorize(norm) != "norm" or len(preds.get(norm.id, [])) != 1:
            continue
        add = graph.nodes[nid]
        if add.kind == "op" and not _in_scope(norm.path, add.path):
            continue  # e.g. last block's residual feeding the model's final norm
        graph.merge({nid, norm.id}, Node(
            id=f"addnorm_{norm.id}", kind="module", type=norm.type, name=norm.name,
            path=norm.path, params=norm.params, order=norm.order,
            label="Add & Norm", detail=module_detail(norm.meta.get("module")),
            category="add_norm",
            meta={"is_add": True, "repeat": norm.meta.get("repeat") or add.meta.get("repeat"),
                  "module": norm.meta.get("module"), "parent": norm.meta.get("parent")}))
        preds, succs = adjacency(graph)


# ----------------------------------------------------------------- labels
INPUT_LABELS = {
    "src": "Inputs", "input_ids": "Inputs", "x": "Input", "input": "Input",
    "inputs": "Inputs", "tgt": "Outputs\n(shifted right)",
    "decoder_input_ids": "Outputs\n(shifted right)", "pixel_values": "Image",
    "inputs_embeds": "Input embeddings",
}


def assign_labels(graph: Graph, architecture: str) -> None:
    """Choose human-friendly titles, detail lines and categories."""
    preds, succs = adjacency(graph)
    for n in graph.nodes.values():
        cat = categorize(n)
        n.category = cat
        module = n.meta.get("module")
        if n.detail is None:
            n.detail = module_detail(module)
        if n.label:
            continue
        if n.kind == "input":
            n.label = INPUT_LABELS.get(n.name, n.name.replace("_", " ").capitalize())
        elif n.kind == "output":
            src = graph.nodes.get(preds[n.id][0]) if preds[n.id] else None
            if src is not None and categorize(src) == "softmax":
                n.label = "Output\nprobabilities"
            else:
                n.label = {"logits": "Logits", "output": "Output"}.get(
                    n.name, n.name.replace("_", " ").capitalize())
        elif n.kind == "param":
            n.label = n.name.rsplit(".", 1)[-1].replace("_", " ").capitalize()
        elif cat == "attention":
            data_in = [p for p in preds[n.id]
                       if categorize(graph.nodes[p]) not in ("positional", "param", "input")
                       or graph.nodes[p].kind == "input" and not re.search("mask", graph.nodes[p].name)]
            data_in = [p for p in data_in if categorize(graph.nodes[p]) not in ("positional", "param")]
            in_decoder = re.search(r"(^|\.)decoder", n.name or "") is not None
            if len(data_in) >= 2:
                n.label = "Multi-Head Cross-Attention"
                n.meta["cross"] = True
            elif n.meta.get("masked") or architecture == "decoder-only" or \
                    (architecture == "encoder-decoder" and in_decoder):
                n.label = "Masked Multi-Head\nAttention"
                n.meta["masked"] = True
            else:
                n.label = "Multi-Head Attention"
        elif cat == "positional" and n.kind == "module" and len(preds[n.id]) == 1 \
                and n.params == 0 and n.meta.get("in_shape") is not None \
                and n.meta.get("in_shape") == n.meta.get("out_shape"):
            # sinusoidal encoding module: draw as ⊕ with a side label
            n.meta["circle"] = "+"
            n.meta["side_label"] = "Positional\nencoding"
            n.label = "+"
        elif cat == "ffn":
            n.label = "Feed Forward"
        elif cat == "norm" and n.type not in ("LayerNorm",) and "rms" in n.type.lower():
            n.label = "RMS Norm"
        elif n.kind == "op" and n.type in ("add",) and _is_positional_add(graph, n, preds):
            n.meta["circle"] = "+"   # the positional source is drawn beside it
            n.label = "+"
        else:
            n.label = friendly_name(n.type)
        if n.kind == "module" and module is not None and n.label != friendly_name(n.type) \
                and not n.meta.get("circle") and n.type not in ("FeedForward", "MultiheadAttention"):
            n.meta.setdefault("class_name", n.type)


def _width_detail(graph: Graph, ids) -> Optional[str]:
    """``d_in → d_hidden → d_out`` from the last dimension of tensor shapes."""
    def width(e):
        return e.shape[-1] if e.shape else None
    w_in = [width(e) for e in graph.edges if e.dst in ids and e.src not in ids]
    w_out = [width(e) for e in graph.edges if e.src in ids and e.dst not in ids]
    w_mid = [width(e) for e in graph.edges if e.src in ids and e.dst in ids]
    w_in, w_out, w_mid = [w for w in w_in if w], [w for w in w_out if w], [w for w in w_mid if w]
    if not (w_in and w_out and w_mid):
        return None
    return f"{w_in[0]} → {max(w_mid)} → {w_out[0]}"


def _is_positional_add(graph, node, preds) -> bool:
    srcs = [graph.nodes[p] for p in preds[node.id]]
    return any(categorize(s) in ("positional", "param") or
               (categorize(s) == "embedding" and re.search(r"pos", s.name, re.I)) for s in srcs)
