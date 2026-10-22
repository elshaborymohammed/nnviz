"""Node categories, display names and colour themes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from torch import nn

from .graph import Node

# (category, regex on the node type) — first match wins.
CATEGORY_RULES = [
    ("positional", r"positional|pos_?enc|posemb|rotary"),
    ("attention", r"attention|attn|scaled_dot_product"),
    ("embedding", r"embed"),
    ("norm", r"norm"),
    ("ffn", r"mlp|feed_?forward|ffn|intermediate"),
    ("softmax", r"softmax"),
    ("recurrent", r"^(lstm|gru|rnn)"),
    ("conv", r"conv(?!1D$)|convolution"),
    ("pool", r"pool"),
    ("linear", r"linear|dense|^conv1D$|bilinear|lm_?head|classifier|proj"),
    ("activation", r"relu|gelu|silu|swish|tanh|sigmoid|elu|mish|softplus|act|glu|hardswish"),
    ("dropout", r"dropout"),
    ("merge", r"^(add|sub|mul|div|cat|stack|matmul|einsum)$"),
]

FRIENDLY_NAMES = {
    "MultiheadAttention": "Multi-Head Attention",
    "LayerNorm": "Layer Norm",
    "RMSNorm": "RMS Norm",
    "BatchNorm1d": "Batch Norm",
    "BatchNorm2d": "Batch Norm",
    "BatchNorm3d": "Batch Norm",
    "GroupNorm": "Group Norm",
    "Conv1d": "Conv 1D",
    "Conv2d": "Conv 2D",
    "Conv3d": "Conv 3D",
    "Conv1D": "Linear",           # HuggingFace GPT-2 style linear layer
    "MaxPool2d": "Max Pool",
    "AvgPool2d": "Avg Pool",
    "AdaptiveAvgPool2d": "Adaptive Avg Pool",
    "softmax": "Softmax",
    "log_softmax": "Log Softmax",
    "scaled_dot_product_attention": "Scaled Dot-Product Attention",
    "linear": "Linear",
    "layer_norm": "Layer Norm",
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "matmul": "MatMul",
    "cat": "Concat",
}

OP_SYMBOLS = {"add": "+", "sub": "−", "mul": "×", "div": "÷", "cat": "‖",
              "matmul": "@", "stack": "≡", "einsum": "Σ"}


def categorize(node: Node) -> str:
    if node.category:
        return node.category
    if node.kind in ("input", "output", "param"):
        return node.kind
    text = node.type
    for cat, pattern in CATEGORY_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return cat
    return "other"


def friendly_name(type_name: str) -> str:
    if type_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[type_name]
    # CamelCase / snake_case → "Title Case"
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", " ", type_name).replace("_", " ")
    return words[:1].upper() + words[1:]


def module_detail(module: Optional[nn.Module]) -> Optional[str]:
    """Short hyper-parameter summary of a module, e.g. ``512 → 2048``."""
    if module is None:
        return None
    m = module
    if isinstance(m, nn.Linear):
        return f"{m.in_features} → {m.out_features}"
    if isinstance(m, nn.Embedding):
        return f"{m.num_embeddings:,} × {m.embedding_dim}"
    if isinstance(m, nn.MultiheadAttention):
        return f"d={m.embed_dim}, heads={m.num_heads}"
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        k = "×".join(str(x) for x in m.kernel_size)
        s = "" if set(m.stride) == {1} else f", s{m.stride[0]}"
        return f"{m.in_channels} → {m.out_channels}, k{k}{s}"
    if isinstance(m, (nn.LayerNorm,)):
        return "d=" + "×".join(str(x) for x in m.normalized_shape)
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return f"c={m.num_features}"
    if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        return f"p={m.p:g}"
    if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
        bi = ", bi" if m.bidirectional else ""
        return f"{m.input_size} → {m.hidden_size}, L={m.num_layers}{bi}"
    w = getattr(m, "weight", None)
    if type(m).__name__ == "Conv1D" and w is not None:   # HF GPT-2
        return f"{w.shape[0]} → {w.shape[1]}"
    heads = next((getattr(m, a) for a in ("num_heads", "n_head", "num_attention_heads",
                                          "n_heads") if isinstance(getattr(m, a, None), int)), None)
    cfg = getattr(m, "config", None)
    if not heads and cfg is not None and "attention" in type(m).__name__.lower():
        heads = getattr(cfg, "num_attention_heads", None)
        kv = getattr(cfg, "num_key_value_heads", None)
        if heads and kv and kv != heads:
            return f"heads={heads}, kv_heads={kv}"
    if heads:
        return f"heads={heads}"
    return None


def human_count(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}".rstrip("0").rstrip(".") + unit
    return str(n)


@dataclass
class Theme:
    """Visual settings. ``fills`` maps category → node fill colour."""
    font: str = "Helvetica"
    font_size: int = 13
    small_font_size: int = 9
    text_color: str = "#111111"
    muted_color: str = "#666666"
    param_color: str = "#2b6f8a"
    extras_color: str = "#8a4b2b"     # activation / dropout line inside a block
    edge_color: str = "#222222"
    residual_color: str = "#222222"
    node_border: str = "#111111"
    background: str = "white"
    repeat_border: str = "#333333"
    repeat_fill: str = "white"
    sublayer_fills: tuple = ("#eef9f6", "#e3f5f0", "#d3efe7")
    sublayer_label_color: str = "#4fbfaa"
    module_fill: str = "#f7f7f7"
    module_border: str = "#bbbbbb"
    fills: Dict[str, str] = field(default_factory=lambda: {"default": "white"})
    sublayer_node_fill: str = "#e9f8f4"

    def fill(self, category: str) -> str:
        return self.fills.get(category, self.fills.get("default", "white"))


# Monochrome style matching the classic "N x" Transformer figure.
PAPER = Theme()

# Colour-coded style in the spirit of the original "Attention Is All You Need" figure.
COLORFUL = Theme(
    sublayer_fills=("#f5f5f5", "#ededed", "#e4e4e4"),
    sublayer_label_color="#888888",
    sublayer_node_fill="",
    fills={
        "default": "#ffffff",
        "attention": "#fbd9a6",
        "ffn": "#c3e6f5",
        "linear": "#dcd3f5",
        "norm": "#f3f4c0",
        "add_norm": "#f3f4c0",
        "embedding": "#f9cfcf",
        "positional": "#ffffff",
        "softmax": "#c8eac9",
        "activation": "#e6f2d9",
        "dropout": "#eeeeee",
        "conv": "#c3e6f5",
        "pool": "#f9e1c7",
        "recurrent": "#fbd9a6",
        "merge": "#ffffff",
    },
)

THEMES = {"paper": PAPER, "color": COLORFUL}
