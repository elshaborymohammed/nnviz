"""Turn an analysed ``Graph`` into Graphviz DOT and PNG / SVG / PDF files."""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .analysis import Residual
from .graph import Edge, Graph, Node
from .style import OP_SYMBOLS, Theme, categorize, human_count


# ------------------------------------------------------------------ clusters
@dataclass
class Cluster:
    id: str
    nodes: Set[str]
    kind: str                  # "module" | "repeat" | "sublayer"
    label: str = ""
    index: int = 0             # sublayer number inside its block
    children: List["Cluster"] = field(default_factory=list)


def build_clusters(graph: Graph, residuals: List[Residual],
                   group_pattern: Optional[str] = r"(^|\.)(encoder|decoder)$",
                   show_repeats: bool = True, show_sublayers: bool = True,
                   sublayer_word: str = "sublayer") -> List[Cluster]:
    """Compute properly nested clusters: module groups ⊇ repeat blocks ⊇ sublayers."""
    def scoped(scope):
        return {n.id for n in graph.nodes.values()
                if n.kind in ("module", "op", "param")
                and (n.path == scope or n.path.startswith(scope + "."))}

    candidates: List[Cluster] = []
    if group_pattern:
        scopes = {_prefix_match(n.path, group_pattern) for n in graph.nodes.values()}
        for scope in sorted(s for s in scopes if s):
            ids = scoped(scope)
            if len(ids) > 1:
                title = scope.rsplit(".", 1)[-1].replace("_", " ").title()
                candidates.append(Cluster(f"group_{scope}", ids, "module", title))
    if show_repeats:
        for scope, info in graph.meta.get("repeats", {}).items():
            ids = scoped(scope)
            if ids:
                label = f"{info['count']} ×"
                candidates.append(Cluster(f"repeat_{scope}", ids, "repeat", label))
    if show_sublayers:
        for r in residuals:
            ids = {n for n in r.span if n in graph.nodes and graph.nodes[n].kind != "input"}
            if len(ids) >= 2:
                candidates.append(Cluster(f"sub_{r.merge}", ids, "sublayer"))

    accepted: List[Cluster] = []
    priority = {"module": 0, "repeat": 1, "sublayer": 2}
    for c in sorted(candidates, key=lambda c: (priority[c.kind], -len(c.nodes))):
        if c.kind == "sublayer":
            # Clip a residual span to the block containing its merge point
            # (shared inputs such as rotary embeddings live outside the block).
            for a in accepted:
                merge = c.id[len("sub_"):]
                if merge in a.nodes and not c.nodes <= a.nodes:
                    c.nodes &= a.nodes
        if all(c.nodes <= a.nodes or a.nodes <= c.nodes or not (c.nodes & a.nodes)
               for a in accepted) and not any(c.nodes == a.nodes and c.kind == a.kind
                                              for a in accepted):
            accepted.append(c)

    # Build the containment tree.
    roots: List[Cluster] = []
    for c in sorted(accepted, key=lambda c: (-len(c.nodes), priority[c.kind])):
        # Smallest enclosing cluster; for identical node sets, the outer kind
        # (module ⊇ repeat ⊇ sublayer) is the parent.
        enclosing = [p for p in accepted if p is not c and (
            c.nodes < p.nodes or (c.nodes == p.nodes and priority[p.kind] < priority[c.kind]))]
        parent = min(enclosing, key=lambda p: (len(p.nodes), -priority[p.kind]), default=None)
        (parent.children if parent else roots).append(c)

    # Number sublayers inside each parent in execution order.
    def number(clusters: List[Cluster]):
        subs = sorted((c for c in clusters if c.kind == "sublayer"),
                      key=lambda c: min(graph.nodes[n].order for n in c.nodes))
        for i, c in enumerate(subs, 1):
            c.index = i
            c.label = _ordinal(i) + "\n" + sublayer_word
        for c in clusters:
            number(c.children)
    number(roots)
    return roots


def _prefix_match(path: str, pattern: str) -> str:
    """Return the shortest prefix of ``path`` that matches ``pattern``."""
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if re.search(pattern, prefix):
            return prefix
    return ""


def _ordinal(i: int) -> str:
    words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"]
    return words[i - 1] if i <= len(words) else f"#{i}"


# ------------------------------------------------------------------ renderer
@dataclass
class RenderOptions:
    direction: str = "BT"              # BT (bottom→top, like the paper), TB, LR, RL
    splines: str = "ortho"             # ortho | spline | polyline | line
    show_params: bool = True
    show_details: bool = True
    show_shapes: bool = False
    show_names: bool = False           # module qualified names under titles
    show_sublayer_labels: bool = True
    residual_junctions: bool = True    # branch residuals from a junction dot
    uniform_width: bool = True         # give every block the width of the widest one
    align_lanes: bool = True           # draw each data path as a straight, centred column
    title: Optional[str] = None
    dpi: int = 150
    nodesep: float = 0.45
    ranksep: float = 0.32


class GraphvizRenderer:
    MIN_BLOCK_WIDTH = 1.9      # inches
    BLOCK_MARGIN = 0.25        # inches of padding left and right of block text
    SYMBOL_SIZE = 0.32         # diameter of ⊕-style operator circles

    def __init__(self, theme: Theme, options: RenderOptions):
        self.theme = theme
        self.opt = options
        self._junction_ids: Set[str] = set()
        self._elbow_ids: Set[str] = set()
        self._lane: Dict[str, str] = {}
        self._block_width = self.MIN_BLOCK_WIDTH

    # -------------------------------------------------------------- DOT text
    def to_dot(self, graph: Graph, clusters: List[Cluster]) -> str:
        t, o = self.theme, self.opt
        lines = [
            "digraph G {",
            f'  graph [rankdir={o.direction}, splines={o.splines}, newrank=true, compound=true,'
            f' nodesep={o.nodesep}, ranksep={o.ranksep}, bgcolor="{t.background}",'
            f' fontname="{t.font}", fontsize={t.font_size}, pad=0.3, dpi={o.dpi}];',
            f'  node [fontname="{t.font}", fontsize={t.font_size}, fontcolor="{t.text_color}",'
            f' color="{t.node_border}", penwidth=1.1];',
            f'  edge [color="{t.edge_color}", arrowsize=0.7, penwidth=1.0, fontname="{t.font}",'
            f' fontsize={t.small_font_size}, fontcolor="{t.muted_color}"];',
        ]
        if o.title:
            lines.append(f'  labelloc=t; label=<{o.title}>;')

        placed: Set[str] = set()
        sublayer_of: Dict[str, Cluster] = {}

        def walk(cs):
            for c in cs:
                if c.kind == "sublayer":
                    for n in c.nodes:
                        sublayer_of[n] = c
                walk(c.children)
        walk(clusters)

        # A residual leaves the main line at a junction just before the
        # sublayer (as in the paper figure) instead of at the previous box.
        junctions: Dict[str, List[str]] = {}
        edges = list(graph.edges)
        if o.residual_junctions:
            edges = self._add_junctions(graph, edges, sublayer_of, junctions)
        self._lane = self._lanes(graph, edges) if o.align_lanes else {}
        elbows: Dict[str, List[tuple]] = {}
        if o.splines in ("ortho", "polyline"):
            edges = self._add_elbows(edges, junctions, elbows)
        self._block_width = self._uniform_width(graph) if o.uniform_width else self.MIN_BLOCK_WIDTH

        def emit_cluster(c: Cluster, indent: str):
            out = [f'{indent}subgraph "cluster_{c.id}" {{']
            inner = indent + "  "
            if c.kind == "repeat":
                out.append(f'{inner}style="dashed,rounded"; color="{t.repeat_border}";'
                           f' penwidth=1.2; bgcolor="{t.repeat_fill}"; margin=14;')
                out.append(f'{inner}label=<<FONT POINT-SIZE="{t.font_size + 1}">{_esc(c.label)}</FONT>>;'
                           f' labeljust=l; labelloc=b;')
            elif c.kind == "sublayer":
                fill = t.sublayer_fills[min(c.index, len(t.sublayer_fills)) - 1] if c.index else t.sublayer_fills[0]
                lbl = (f'<FONT POINT-SIZE="{t.font_size - 1}" COLOR="{t.sublayer_label_color}">'
                       f'{_lines(c.label)}</FONT>') if o.show_sublayer_labels else '""'
                out.append(f'{inner}style="filled,rounded"; color="{fill}"; fillcolor="{fill}";'
                           f' margin=10; label=<{lbl}>; labeljust=r; labelloc=t;')
            else:
                out.append(f'{inner}style="rounded"; color="{t.module_border}"; penwidth=0.8;'
                           f' bgcolor="transparent"; margin=16; labeljust=c; labelloc=t;'
                           f' label=<<FONT COLOR="{t.muted_color}">{_esc(c.label)}</FONT>>;')
            child_nodes = set().union(*(ch.nodes for ch in c.children)) if c.children else set()
            for ch in c.children:
                out.extend(emit_cluster(ch, inner))
            for jid in junctions.get(c.id, []):
                out.append(inner + _stmt(jid, self._lane_attrs(jid), shape="point", width=0.06,
                                         color=t.edge_color))
            for eid, merge in elbows.get(c.id, []):
                out.append(inner + _stmt(eid, {}, shape="point", width=0.01, style="invis"))
                out.append(f'{inner}{{rank=same; "{merge}"; "{eid}";}}')
            for nid in sorted(c.nodes - child_nodes, key=lambda n: graph.nodes[n].order):
                if nid not in placed:
                    placed.add(nid)
                    out.append(inner + self._node(graph.nodes[nid], sublayer_of.get(nid)))
            out.append(indent + "}")
            return out

        for c in clusters:
            lines.extend(emit_cluster(c, "  "))
        for n in graph.sorted_nodes():
            if n.id not in placed:
                lines.append("  " + self._node(n, None))
                placed.add(n.id)

        # Side labels for ⊕ nodes (e.g. "Positional encoding").
        for n in graph.sorted_nodes():
            side = n.meta.get("side_label")
            if side:
                sid = f"{n.id}__side"
                lines.append(f'  "{sid}" [shape=plaintext, label=<{_lines(side)}>];')
                lines.append(f'  {{rank=same; "{sid}"; "{n.id}";}}')
                lines.append(f'  "{sid}" -> "{n.id}" [style=invis];')

        aux = self._junction_ids | self._elbow_ids
        for e in edges:
            if e.src not in graph.nodes and e.src not in aux or e.dst not in graph.nodes and e.dst not in aux:
                continue
            attrs = []
            if e.dst in aux:
                attrs.append("arrowhead=none")
            if e.dst in self._elbow_ids:    # run both halves to the elbow's centre so they join
                attrs.append("headclip=false")
            if e.kind == "elbow":       # drawn merge → elbow, arrow pointing back at the merge
                attrs += ["dir=back", f'color="{t.residual_color}"']
            if e.kind == "residual":
                attrs.append(f'color="{t.residual_color}"')
                if o.align_lanes:   # skip edges must not pull blocks off their lane
                    attrs.append("weight=0")
                if o.splines == "polyline" and e.dst not in self._elbow_ids:   # ortho ignores ports
                    attrs += ['tailport=e', 'headport=e'] if o.direction in ("BT", "TB") else \
                             ['tailport=s', 'headport=s']
            if o.show_shapes and e.shape:
                attrs.append(f'xlabel="{"×".join(str(s) for s in e.shape)}"')
            lines.append(f'  "{e.src}" -> "{e.dst}"' + (f' [{", ".join(attrs)}]' if attrs else "") + ";")
        lines.append("}")
        return "\n".join(lines)

    def _add_junctions(self, graph: Graph, edges: List[Edge],
                       sublayer_of: Dict[str, Cluster], junctions: Dict[str, List[str]]) -> List[Edge]:
        out = []
        rerouted: Dict[tuple, str] = {}
        for e in edges:
            if e.kind != "residual" or e.dst not in sublayer_of:
                continue
            sub = sublayer_of[e.dst]
            jid = f"junction_{e.dst}"
            targets = [d for d in edges if d.src == e.src and d.kind == "data" and d.dst in sub.nodes]
            if not targets:
                continue
            junctions.setdefault(sub.id, []).append(jid)
            self._junction_ids.add(jid)
            rerouted[(e.src, e.dst)] = jid
            for d in targets:
                rerouted[(d.src, d.dst)] = jid
        added = set()
        for e in edges:
            jid = rerouted.get((e.src, e.dst))
            if jid is None:
                out.append(e)
                continue
            if jid not in added:
                added.add(jid)
                out.append(Edge(e.src, jid, e.shape, "data"))
            out.append(Edge(jid, e.dst, e.shape, e.kind))
        return out

    def _add_elbows(self, edges: List[Edge], junctions: Dict[str, List[str]],
                    elbows: Dict[str, List[tuple]]) -> List[Edge]:
        """Route each junction → merge residual through an invisible elbow point.

        Orthogonal splines ignore edge ports, so a residual can end up drawn on
        either side of its lane, sometimes straight through the blocks.  An elbow
        placed in the merge node's rank, just right of it (a flat ``merge → elbow``
        edge orders the two), pins the route: up the right-hand side, then into
        the merge from the right.
        """
        cluster_of = {jid: cid for cid, jids in junctions.items() for jid in jids}
        out = []
        for e in edges:
            if e.kind != "residual" or e.src not in cluster_of:
                out.append(e)
                continue
            eid = f"elbow_{e.src}__{e.dst}"
            self._elbow_ids.add(eid)
            elbows.setdefault(cluster_of[e.src], []).append((eid, e.dst))
            out += [Edge(e.src, eid, e.shape, "residual"), Edge(e.dst, eid, None, "elbow")]
        return out

    # ----------------------------------------------------------------- lanes
    def _lanes(self, graph: Graph, edges: List[Edge]) -> Dict[str, str]:
        """Assign each node to a vertical lane (a Graphviz ``group``).

        Graphviz keeps edges inside a group straight, so every lane is drawn as a
        centred column.  Lanes are traced backwards from the outputs: each node
        continues the lane through its predecessor in the closest module scope
        (the decoder, not the encoder memory, for cross-attention), then the one
        with the longest chain behind it (the token embedding, not a learned
        positional parameter).  Nodes left over, such as projection shortcuts or
        the encoder, start new lanes.
        """
        preds: Dict[str, List[str]] = defaultdict(list)
        for e in edges:
            if e.kind != "residual":
                preds[e.dst].append(e.src)

        def node_of(nid: str) -> Node:      # junctions stand in for their target
            return graph.nodes[nid] if nid in graph.nodes else graph.nodes[nid[len("junction_"):]]

        def shared_scope(a: str, b: str) -> int:
            pa, pb = (node_of(x).path.split(".") if node_of(x).path else [] for x in (a, b))
            return next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))

        ids = list(graph.nodes) + sorted(self._junction_ids)
        depth: Dict[str, int] = {}          # longest chain of predecessors
        for nid in sorted(ids, key=lambda i: (node_of(i).order, i not in self._junction_ids)):
            depth[nid] = 1 + max((depth.get(p, 0) for p in preds[nid]), default=0)

        starts = sorted(ids, key=lambda i: (node_of(i).kind != "output", -node_of(i).order))
        lane: Dict[str, str] = {}
        for start in starts:
            if start in lane:
                continue
            name, cur = f"lane{len(set(lane.values()))}", start
            while cur is not None:
                lane[cur] = name
                free = [p for p in preds[cur] if p not in lane]
                cur = max(free, key=lambda p: (shared_scope(p, cur), depth[p], node_of(p).order),
                          default=None)
        return lane

    # ----------------------------------------------------------------- nodes
    def _node(self, n: Node, sublayer: Optional[Cluster]) -> str:
        if self._is_symbol(n):
            return self._symbol_node(n)
        if n.kind in ("input", "output"):
            return _stmt(n.id, self._lane_attrs(n.id), shape="plaintext", label=f"<{_lines(n.title)}>")
        if n.kind == "param":
            return self._param_node(n)
        return self._block_node(n, sublayer)

    @staticmethod
    def _is_symbol(n: Node) -> bool:
        return bool(n.meta.get("circle")) or (n.kind == "op" and n.type in OP_SYMBOLS)

    def _is_block(self, n: Node) -> bool:
        return not self._is_symbol(n) and n.kind not in ("input", "output", "param")

    def _lane_attrs(self, nid: str) -> Dict[str, str]:
        return {"group": self._lane[nid]} if nid in self._lane else {}

    def _symbol_node(self, n: Node) -> str:
        sym = n.meta.get("circle") or OP_SYMBOLS[n.type]
        return _stmt(n.id, self._lane_attrs(n.id), shape="circle", fixedsize="true",
                     width=self.SYMBOL_SIZE, label=sym, fontsize=self.theme.font_size + 2,
                     style="filled", fillcolor="white")

    def _param_node(self, n: Node) -> str:
        t = self.theme
        shape = "×".join(str(s) for s in n.meta.get("shape", ()))
        label = f'{_lines(n.title)}<BR/>{_font(shape, t.small_font_size, t.muted_color)}'
        if self.opt.show_params and n.params:
            label += "<BR/>" + _font(f"{human_count(n.params)} params", t.small_font_size, t.param_color)
        return _stmt(n.id, self._lane_attrs(n.id), shape="plaintext", label=f"<{label}>")

    def _block_rows(self, n: Node) -> List[tuple]:
        """The text lines of a block as ``(text, font_size, colour)``."""
        t, o = self.theme, self.opt
        rows = [(line, t.font_size, None) for line in str(n.title).split("\n")]
        if n.meta.get("class_name") and n.meta["class_name"] != n.title:
            rows.append((n.meta["class_name"], t.small_font_size, t.muted_color))
        if o.show_names and n.name:
            rows.append((n.name, t.small_font_size, t.muted_color))
        if o.show_details and n.detail:
            rows.append((n.detail, t.small_font_size, t.muted_color))
        if o.show_params and n.params:
            rows.append((f"{human_count(n.params)} params", t.small_font_size, t.param_color))
        return rows

    def _block_node(self, n: Node, sublayer: Optional[Cluster]) -> str:
        t = self.theme
        cells = "".join(f"<TR><TD>{_font(text, size if size != t.font_size else None, color)}</TD></TR>"
                        for text, size, color in self._block_rows(n))
        label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">{cells}</TABLE>>'
        fill = t.fill(categorize(n))
        if sublayer is not None and t.sublayer_node_fill and fill == t.fill("default"):
            fill = t.sublayer_node_fill
        return _stmt(n.id, self._lane_attrs(n.id), shape="box", style="rounded,filled",
                     fillcolor=fill, margin=f"{self.BLOCK_MARGIN},0.06",
                     width=f"{self._block_width:.2f}", label=label)

    def _uniform_width(self, graph: Graph) -> float:
        """Width (inches) that fits the widest block, so every block lines up."""
        widest = max((_text_width(text, size) for n in graph.nodes.values() if self._is_block(n)
                      for text, size, _ in self._block_rows(n)), default=0.0)
        return max(self.MIN_BLOCK_WIDTH, widest + 2 * self.BLOCK_MARGIN)

    # ----------------------------------------------------------------- output
    def render(self, dot: str, path: str) -> Path:
        path = Path(path)
        fmt = path.suffix.lstrip(".").lower() or "png"
        if fmt == "dot":
            path.write_text(dot)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.render_bytes(dot, fmt))
        return path

    def render_bytes(self, dot: str, fmt: str = "png") -> bytes:
        """Render in memory, without touching the filesystem."""
        if fmt == "dot":
            return dot.encode()
        exe = shutil.which("dot")
        if exe is None:
            raise RuntimeError("Graphviz is not installed: install it (e.g. `brew install graphviz`"
                               " or `apt install graphviz`), or save as .dot")
        proc = subprocess.run([exe, f"-T{fmt}"], input=dot.encode(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Graphviz failed: {proc.stderr.decode(errors='replace')}")
        return proc.stdout


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _lines(text: str) -> str:
    return "<BR/>".join(_esc(s) for s in str(text).split("\n"))


def _font(text: str, size: Optional[int] = None, color: Optional[str] = None) -> str:
    attrs = (f' POINT-SIZE="{size}"' if size else "") + (f' COLOR="{color}"' if color else "")
    return f"<FONT{attrs}>{_esc(text)}</FONT>" if attrs else _esc(text)


def _text_width(text: str, font_size: int) -> float:
    """Rough rendered width in inches (Helvetica averages ~0.58 em per glyph)."""
    return len(text) * font_size * 0.58 / 72


def _stmt(nid: str, extra: Dict[str, str], **attrs) -> str:
    """A DOT node statement; values wrapped in ``<…>`` are HTML labels."""
    attrs.update(extra)
    body = ", ".join(f"{k}={v}" if str(v).startswith("<") or isinstance(v, (int, float))
                     else f'{k}="{_esc(v)}"' for k, v in attrs.items())
    return f'"{nid}" [{body}];'
