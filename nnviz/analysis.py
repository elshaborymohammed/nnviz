"""Graph analysis: residual / skip connections and architecture detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from torch import nn

from .graph import Graph
from .style import categorize


@dataclass
class Residual:
    """A skip connection: ``source`` bypasses ``branch`` and joins at ``merge``."""
    source: str
    merge: str
    branch: Set[str] = field(default_factory=set)
    shortcut: Set[str] = field(default_factory=set)  # projection shortcut nodes

    @property
    def span(self) -> Set[str]:
        return self.branch | self.shortcut | {self.merge}


def adjacency(graph: Graph):
    preds: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    succs: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        if e.src in graph.nodes and e.dst in graph.nodes:
            preds[e.dst].append(e.src)
            succs[e.src].append(e.dst)
    return preds, succs


def _closure(start: str, nxt: Dict[str, List[str]], limit: Optional[Set[str]] = None) -> Set[str]:
    seen, stack = set(), [start]
    while stack:
        for n in nxt[stack.pop()]:
            if n not in seen and (limit is None or n in limit):
                seen.add(n)
                stack.append(n)
    return seen


def _has_layer(graph: Graph, ids: Set[str]) -> bool:
    """A real skip connection bypasses at least one learned layer."""
    return any(graph.nodes[i].params > 0 and graph.nodes[i].kind == "module" for i in ids)


def is_merge_add(graph: Graph, nid: str) -> bool:
    n = graph.nodes[nid]
    return (n.kind == "op" and n.type in ("add", "sub")) or n.meta.get("is_add", False)


def find_residuals(graph: Graph, mark_edges: bool = True) -> List[Residual]:
    """Detect skip connections and (optionally) mark their edges ``residual``.

    An addition ``y = a + b`` is residual when ``a`` is an ancestor of ``b``
    (identity shortcut), or when ``a`` and ``b`` fork from a common node
    (projection shortcut, e.g. ResNet downsampling) — the shorter path is the
    shortcut.
    """
    preds, succs = adjacency(graph)
    residuals: List[Residual] = []
    anc_cache: Dict[str, Set[str]] = {}

    def anc(n):
        if n not in anc_cache:
            anc_cache[n] = _closure(n, preds)
        return anc_cache[n]

    for e in graph.edges:
        if e.kind == "residual":
            e.kind = "data"

    for nid in sorted(graph.nodes, key=lambda k: graph.nodes[k].order):
        if not is_merge_add(graph, nid):
            continue
        ins = [p for p in preds[nid] if graph.nodes[p].kind not in ("param",)]
        if len(ins) < 2:
            continue
        found = None
        for a in ins:
            for b in ins:
                if a != b and a in anc(b):
                    region = anc(b) | {b}
                    branch = _closure(a, succs, region)
                    if not _has_layer(graph, branch):
                        continue
                    found = Residual(a, nid, branch)
                    skip_edges = [(a, nid)]
                    break
            if found:
                break
        if not found and len(ins) == 2:
            a, b = ins
            common = (anc(a) | {a}) & (anc(b) | {b})
            common = {c for c in common if graph.nodes[c].kind != "param"}
            if common:
                fork = max(common, key=lambda c: graph.nodes[c].order)
                path_a = _closure(fork, succs, anc(a) | {a})
                path_b = _closure(fork, succs, anc(b) | {b})
                if len(path_a) > len(path_b):
                    path_a, path_b, a, b = path_b, path_a, b, a
                if _has_layer(graph, path_b):
                    found = Residual(fork, nid, path_b, path_a)
                    skip_edges = [(a, nid)] if path_a else [(fork, nid)]
        if found:
            residuals.append(found)
            if mark_edges:
                for e in graph.edges:
                    if (e.src, e.dst) in skip_edges:
                        e.kind = "residual"
    return residuals


# --------------------------------------------------------------- architecture
_DECODER_ONLY_TYPES = {
    "gpt2", "gpt_neo", "gpt_neox", "gptj", "llama", "mistral", "mixtral", "qwen2",
    "qwen3", "phi", "phi3", "gemma", "gemma2", "gemma3", "falcon", "opt", "bloom",
    "codegen", "olmo", "starcoder2", "mpt", "gpt_bigcode", "stablelm", "cohere",
}


def detect_architecture(model: nn.Module, graph: Optional[Graph] = None) -> str:
    """Return ``encoder-decoder``, ``decoder-only``, ``encoder-only`` or ``generic``."""
    config = getattr(model, "config", None)
    if config is not None:
        if getattr(config, "is_encoder_decoder", False):
            return "encoder-decoder"
        archs = " ".join(getattr(config, "architectures", None) or [])
        if getattr(config, "is_decoder", False) or "CausalLM" in archs or "LMHead" in archs \
                or getattr(config, "model_type", "") in _DECODER_ONLY_TYPES:
            return "decoder-only"
    if isinstance(model, nn.Transformer):
        return "encoder-decoder"

    names = [n.lower() for n, _ in model.named_modules()]
    has_enc = any(re.search(r"(^|\.)encoder", n) for n in names)
    has_dec = any(re.search(r"(^|\.)decoder", n) for n in names)
    has_attn = graph is not None and any(categorize(n) == "attention" for n in graph.nodes.values())
    if not has_attn:
        has_attn = any(isinstance(m, nn.MultiheadAttention) or "attention" in type(m).__name__.lower()
                       for m in model.modules())
    if not has_attn:
        return "generic"
    if has_enc and has_dec:
        return "encoder-decoder"
    if graph is not None and any(n.meta.get("masked") for n in graph.nodes.values()):
        return "decoder-only"
    if has_dec:
        return "decoder-only"
    return "encoder-only"
