"""High-level entry point: ``NNVisualizer(model).draw("model.png")``."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union

from torch import nn

from . import transforms as T
from .analysis import detect_architecture, find_residuals
from .graph import Graph
from .inputs import make_dummy_input
from .render import GraphvizRenderer, RenderOptions, build_clusters
from .style import THEMES, Theme, human_count
from .tracer import Tracer

Pass = Callable[[Graph], None]


class NNVisualizer:
    """Draw any PyTorch model as a block diagram, Transformer-paper style.

    Pipeline (each stage is replaceable): trace → transform passes → residual
    detection → clustering → Graphviz rendering.

    Args:
        model: a raw ``nn.Module`` (custom, ``torch.nn``, HuggingFace, ...).
        input_data: example input — tensor, tuple of positional args or dict
            of kwargs.  Guessed automatically when omitted.
        theme: ``"paper"`` (monochrome, like the classic figure), ``"color"``
            or a :class:`Theme` instance.
        collapse: regex on module class names to draw as single boxes,
            or a callable ``(qualified_name, module) -> bool``.
        max_depth: additionally collapse every module at this nesting depth
            (``1`` = only top-level children).
        collapse_repeats: fold identical stacked layers into one ``N ×`` block.
        merge_ffn: fold ``Linear → act → Linear`` residual branches into
            ``Feed Forward``.
        merge_add_norm: fuse ``x + sublayer(x) → LayerNorm`` into ``Add & Norm``.
        hide_dropout: drop Dropout layers (Identity layers are always dropped).
        hide_inputs: regex of input names not to draw (masks by default).
        group_pattern: regex of module names drawn as labelled groups
            (encoder / decoder by default; ``None`` to disable).
        leaf_types: extra module classes to treat as atomic when tracing.
        extra_passes: callables ``f(graph)`` run after the built-in passes.
        **render_options: any field of :class:`RenderOptions`
            (``direction``, ``splines``, ``show_params``, ``show_shapes``, ...).
    """

    def __init__(self, model: nn.Module, input_data: Any = None, *,
                 theme: Union[str, Theme] = "paper",
                 collapse: Union[str, Callable, None] = None,
                 max_depth: Optional[int] = None,
                 collapse_repeats: bool = True,
                 merge_ffn: bool = True,
                 merge_add_norm: bool = True,
                 hide_dropout: bool = False,
                 hide_inputs: Optional[str] = r"mask",
                 group_pattern: Optional[str] = r"(^|\.)(encoder|decoder)$",
                 show_sublayers: bool = True,
                 leaf_types: Iterable[type] = (),
                 extra_passes: Iterable[Pass] = (),
                 **render_options):
        self.model = model
        self.input_data = input_data
        self.theme = THEMES[theme] if isinstance(theme, str) else theme
        if collapse is None:
            self.collapse_rule = T.default_collapse_rule()
        elif isinstance(collapse, str):
            self.collapse_rule = T.default_collapse_rule(collapse)
        else:
            self.collapse_rule = collapse
        self.max_depth = max_depth
        self.collapse_repeats = collapse_repeats
        self.merge_ffn = merge_ffn
        self.merge_add_norm = merge_add_norm
        self.hide_dropout = hide_dropout
        self.hide_inputs = hide_inputs
        self.group_pattern = group_pattern
        self.show_sublayers = show_sublayers
        self.leaf_types = tuple(leaf_types)
        self.extra_passes = list(extra_passes)
        self.options = RenderOptions(**render_options)
        self.graph: Optional[Graph] = None
        self.architecture: Optional[str] = None

    # ------------------------------------------------------------ pipeline
    def trace(self) -> Graph:
        data = self.input_data if self.input_data is not None else make_dummy_input(self.model)
        return Tracer(self.model, leaf_types=self.leaf_types).trace(data)

    def passes(self) -> List[Pass]:
        """The ordered list of graph rewrites; override to customise."""
        m = self.model
        ps: List[Pass] = [T.record_attention_flags,
                          lambda g: T.hide_inputs(g, self.hide_inputs)]
        if self.hide_dropout:
            ps.append(T.drop_nodes)
        else:
            ps.append(lambda g: T.drop_nodes(g, types=("Identity",), categories=()))
        ps.append(lambda g: T.collapse_modules(g, m, self.collapse_rule, self.max_depth))
        if self.collapse_repeats:
            ps.append(lambda g: T.collapse_repeats(g, m))
        if self.merge_ffn:
            ps.append(T.merge_feed_forward)
        if self.merge_add_norm:
            ps.append(T.merge_add_norm)
        ps.extend(self.extra_passes)
        return ps

    def build(self) -> Graph:
        graph = self.trace()
        for p in self.passes():
            p(graph)
        graph.prune_to_outputs()
        self.architecture = detect_architecture(self.model, graph)
        T.assign_labels(graph, self.architecture)
        self.residuals = find_residuals(graph)
        self.graph = graph
        return graph

    def _title(self, graph: Graph) -> str:
        total, trainable = graph.meta["total_params"], graph.meta["trainable_params"]
        arch = {"encoder-decoder": "Encoder–Decoder Transformer",
                "decoder-only": "Decoder-only Transformer",
                "encoder-only": "Encoder-only Transformer",
                "generic": "Neural Network"}[self.architecture]
        sub = f"{human_count(total)} parameters"
        if trainable != total:
            sub += f" ({human_count(trainable)} trainable)"
        return (f'<FONT POINT-SIZE="{self.theme.font_size + 5}"><B>{graph.meta["model_class"]}</B></FONT>'
                f'<BR/><FONT POINT-SIZE="{self.theme.font_size - 1}" COLOR="{self.theme.muted_color}">'
                f'{arch} · {sub}</FONT><BR/> ')

    def to_dot(self) -> str:
        graph = self.graph or self.build()
        clusters = build_clusters(graph, self.residuals, self.group_pattern,
                                  self.collapse_repeats, self.show_sublayers,
                                  "residual block" if self.architecture == "generic" else "sublayer")
        if self.options.title is None:
            self.options.title = self._title(graph)
        return GraphvizRenderer(self.theme, self.options).to_dot(graph, clusters)

    def draw(self, path: Union[str, Path] = "model.png") -> Path:
        """Render to ``.png``, ``.svg``, ``.pdf`` or ``.dot`` (by extension)."""
        return GraphvizRenderer(self.theme, self.options).render(self.to_dot(), str(path))

    def to_bytes(self, fmt: str = "png") -> bytes:
        """Render to ``png`` / ``svg`` / ``pdf`` bytes in memory (no file written)."""
        return GraphvizRenderer(self.theme, self.options).render_bytes(self.to_dot(), fmt)

    def plot(self, ax=None, figsize=None, show: bool = True):
        """Display the diagram without saving it.

        Inside Jupyter / IPython kernels it is shown inline; elsewhere it is drawn
        with matplotlib (on ``ax`` if given, otherwise a new figure sized to the
        image, capped at 10 inches tall).  Returns the matplotlib axes, or
        ``None`` when shown inline.
        """
        png = self.to_bytes("png")
        if ax is None and _in_notebook():
            from IPython.display import Image, display
            display(Image(data=png))
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plot() outside Jupyter needs matplotlib: `pip install matplotlib`"
                              " (or use draw() to save a file)") from None
        img = plt.imread(io.BytesIO(png), format="png")
        if ax is None:
            h, w = img.shape[:2]
            dpi = self.options.dpi
            scale = min(1.0, 10 / (h / dpi))
            fig, ax = plt.subplots(figsize=figsize or (w / dpi * scale, h / dpi * scale), dpi=dpi)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(img)
        ax.axis("off")
        if show:
            plt.show()
        return ax

    def _repr_png_(self) -> bytes:
        """Lets a bare ``NNVisualizer(...)`` at the end of a notebook cell render itself."""
        return self.to_bytes("png")

    def summary(self) -> str:
        """Text overview of the drawn blocks."""
        graph = self.graph or self.build()
        rows = [f"{graph.meta['model_class']} ({self.architecture}), "
                f"{human_count(graph.meta['total_params'])} params"]
        for scope, info in graph.meta.get("repeats", {}).items():
            rows.append(f"  {info['count']}× {info['type']} @ {scope} "
                        f"({human_count(info['params'])} params each)")
        rows.append(f"  {len(graph.nodes)} blocks, {len(self.residuals)} residual connections")
        return "\n".join(rows)


def draw_model(model: nn.Module, path: Union[str, Path] = "model.png",
               input_data: Any = None, **kwargs) -> Path:
    """One-liner: ``draw_model(model, "model.svg")``."""
    return NNVisualizer(model, input_data, **kwargs).draw(path)


def plot_model(model: nn.Module, input_data: Any = None, ax=None, **kwargs):
    """One-liner: ``plot_model(model)`` shows the diagram without saving it."""
    return NNVisualizer(model, input_data, **kwargs).plot(ax=ax)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config
