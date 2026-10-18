"""Lightweight dataflow graph used by every stage of nnviz.

A traced model becomes a ``Graph`` of ``Node`` objects (modules, functional
ops, model inputs/outputs, free parameters) connected by ``Edge`` objects that
carry tensor shapes.  Transform passes rewrite this graph; renderers draw it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class Node:
    id: str
    kind: str                      # "module" | "op" | "input" | "output" | "param"
    type: str                      # class name ("Linear") or op name ("add")
    name: str = ""                 # qualified module name, arg name, ...
    path: str = ""                 # qualified name of the owning module scope
    params: int = 0                # parameter count represented by this node
    order: int = 0                 # execution order (for stable layouts)
    label: Optional[str] = None    # display title (set by transforms)
    detail: Optional[str] = None   # extra display line (e.g. "512 → 2048")
    category: Optional[str] = None # style category (attention, ffn, norm, ...)
    meta: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.label or self.type


@dataclass
class Edge:
    src: str
    dst: str
    shape: Optional[Tuple[int, ...]] = None
    kind: str = "data"             # "data" | "residual"


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        # Extra information collected while tracing / transforming.
        self.meta: dict = {}

    # ------------------------------------------------------------------ basics
    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_edge(self, src: str, dst: str, shape=None, kind: str = "data") -> None:
        if src == dst:
            return
        for e in self.edges:
            if e.src == src and e.dst == dst:
                if e.shape is None:
                    e.shape = shape
                return
        self.edges.append(Edge(src, dst, shape, kind))

    def in_edges(self, nid: str) -> List[Edge]:
        return [e for e in self.edges if e.dst == nid]

    def out_edges(self, nid: str) -> List[Edge]:
        return [e for e in self.edges if e.src == nid]

    def preds(self, nid: str) -> List[str]:
        return [e.src for e in self.in_edges(nid)]

    def succs(self, nid: str) -> List[str]:
        return [e.dst for e in self.out_edges(nid)]

    def sorted_nodes(self) -> List[Node]:
        return sorted(self.nodes.values(), key=lambda n: n.order)

    # ----------------------------------------------------------- rewriting
    def remove_node(self, nid: str) -> None:
        self.nodes.pop(nid, None)
        self.edges = [e for e in self.edges if e.src != nid and e.dst != nid]

    def dissolve(self, nid: str) -> None:
        """Remove a node, wiring each predecessor to each successor."""
        ins, outs = self.in_edges(nid), self.out_edges(nid)
        self.remove_node(nid)
        for i in ins:
            for o in outs:
                self.add_edge(i.src, o.dst, o.shape, o.kind)

    def merge(self, nids: Iterable[str], new: Node) -> Node:
        """Replace a set of nodes by a single node, keeping external edges."""
        group = set(nids)
        ext_in, ext_out = [], []
        for e in self.edges:
            if e.dst in group and e.src not in group:
                ext_in.append(e)
            elif e.src in group and e.dst not in group:
                ext_out.append(e)
        self.edges = [e for e in self.edges if e.src not in group and e.dst not in group]
        for nid in group:
            self.nodes.pop(nid, None)
        self.add_node(new)
        for e in ext_in:
            self.add_edge(e.src, new.id, e.shape, e.kind)
        for e in ext_out:
            self.add_edge(new.id, e.dst, e.shape, e.kind)
        return new

    # ----------------------------------------------------------- analysis
    def ancestors(self, nid: str) -> Set[str]:
        seen, stack = set(), [nid]
        while stack:
            for p in self.preds(stack.pop()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def descendants(self, nid: str) -> Set[str]:
        seen, stack = set(), [nid]
        while stack:
            for s in self.succs(stack.pop()):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return seen

    def prune_to_outputs(self) -> None:
        """Drop everything that does not contribute to an output node."""
        outputs = [n.id for n in self.nodes.values() if n.kind == "output"]
        keep = set(outputs)
        for o in outputs:
            keep |= self.ancestors(o)
        for nid in list(self.nodes):
            if nid not in keep:
                self.remove_node(nid)
