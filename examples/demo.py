"""Render a gallery of models with nnviz.

    .venv/bin/python examples/demo.py          # writes PNG + SVG files to out/
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnviz import NNVisualizer  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


# ----------------------------------------------------------- building blocks
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                      # x: (B, T, D)
        return x + self.pe[: x.size(1)]


class Seq2SeqTransformer(nn.Module):
    """The original encoder–decoder Transformer (Vaswani et al., 2017)."""

    def __init__(self, vocab=1000, d_model=128, nhead=4, layers=6, d_ff=512):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab, d_model)
        self.tgt_embedding = nn.Embedding(vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, layers, layers, d_ff, batch_first=True)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, src, tgt):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        memory = self.transformer.encoder(self.pos_encoding(self.src_embedding(src)))
        out = self.transformer.decoder(self.pos_encoding(self.tgt_embedding(tgt)), memory,
                                       tgt_mask=mask, tgt_is_causal=True)
        return F.softmax(self.linear(out), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads, self.qkv, self.proj = heads, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).transpose(1, 3).unbind(2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class GPTBlock(nn.Module):
    """Pre-norm decoder block: x + attn(ln(x)), x + mlp(ln(x))."""

    def __init__(self, d, heads):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(d), CausalSelfAttention(d, heads)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab=1000, d=128, heads=4, layers=6, ctx=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, ctx, d))
        self.blocks = nn.ModuleList(GPTBlock(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


class MiniBERT(nn.Module):
    def __init__(self, vocab=1000, d=128, heads=4, layers=4, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.pos_encoding = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d, classes)

    def forward(self, ids):
        h = self.encoder(self.pos_encoding(self.embedding(ids)))
        return self.classifier(h[:, 0])


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (self.down(x) if self.down is not None else x))


class TinyResNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.stage1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32), BasicBlock(32, 32))
        self.stage2 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, classes)

    def forward(self, x):
        return self.fc(self.pool(self.stage2(self.stage1(self.stem(x)))).flatten(1))


def main():
    OUT.mkdir(exist_ok=True)
    src = torch.randint(0, 1000, (1, 10))
    tgt = torch.randint(0, 1000, (1, 9))
    jobs = [
        # ("seq2seq_transformer", Seq2SeqTransformer(), (src, tgt), {}),
        # ("seq2seq_transformer_color", Seq2SeqTransformer(), (src, tgt), {"theme": "color"}),
        # ("pytorch_nn_transformer", nn.Transformer(64, 4, 3, 3, 256), None, {}),
        ("mini_gpt", MiniGPT(), src, {}),
        # ("mini_bert", MiniBERT(), src, {"theme": "color"}),
        # ("tiny_resnet", TinyResNet(), torch.randn(1, 3, 32, 32), {"theme": "color"}),
        ("mlp", nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                              nn.Linear(64, 10)), None, {}),
    ]
    for name, model, inputs, kwargs in jobs:
        viz = NNVisualizer(model, inputs, **kwargs)
        print(viz.summary())
        for ext in ("png", "svg"):
            print("  ->", viz.draw(OUT / f"{name}.{ext}"))


if __name__ == "__main__":
    main()
