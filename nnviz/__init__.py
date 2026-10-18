"""nnviz — draw PyTorch models (Transformers and beyond) as block diagrams.

    from nnviz import NNVisualizer, draw_model
    draw_model(model, "model.svg")                 # one-liner
    NNVisualizer(model, inputs, theme="color").draw("model.png")
"""
from .analysis import detect_architecture, find_residuals
from .graph import Edge, Graph, Node
from .render import GraphvizRenderer, RenderOptions
from .style import COLORFUL, PAPER, THEMES, Theme
from .tracer import Tracer
from .visualizer import NNVisualizer, draw_model, plot_model

__all__ = ["NNVisualizer", "draw_model", "plot_model", "Tracer", "Graph", "Node", "Edge",
           "GraphvizRenderer", "RenderOptions", "Theme", "THEMES", "PAPER", "COLORFUL",
           "find_residuals", "detect_architecture"]
