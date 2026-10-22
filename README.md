# nnviz

Draw any PyTorch model as a block diagram in the style of the classic Transformer
figure: bottom-to-top flow, `N ×` stacks, shaded sublayers, residual arrows,
`Add & Norm`, a ⊕ for positional encoding, and parameter counts on every block.

```python
from nnviz import NNVisualizer, draw_model, plot_model

draw_model(model, "model.svg")                         # input is guessed
NNVisualizer(model, (src, tgt), theme="color").draw("model.png")

plot_model(model)                                      # show it, no file written
v = NNVisualizer(model, theme="color")
v.plot()                                               # inline in Jupyter, else a matplotlib window
v.plot(ax=axes[0])                                     # onto an existing matplotlib axes
png = v.to_bytes("png")                                # raw bytes: "png", "svg" or "pdf"
```

In a notebook, a bare `v` at the end of a cell also renders itself. Outside
Jupyter, `plot()` needs matplotlib (`pip install "nnviz[plot]"`).

Run `examples/demo.py` (custom models) and `examples/hf_demo.py` (BERT, GPT-2,
T5, Llama, ViT) to render the gallery into `out/`.

## How it works

| Module | Role |
|---|---|
| `tracer.py` | Runs one real forward pass with module hooks plus a `TorchFunctionMode`, so functional ops such as `x + sublayer(x)` (residuals), `torch.cat` and `F.softmax` are captured. Works with Python control flow and HuggingFace models. |
| `transforms.py` | Rewrite passes: drop Dropout/Identity, collapse modules (attention, MLP, embeddings), fold `layers.0…N` into `N ×`, merge `Linear→act→Linear` into **Feed Forward**, fuse `add→LayerNorm` into **Add & Norm**, assign labels (masked / cross attention). |
| `analysis.py` | Residual detection (identity and projection shortcuts) and architecture detection (encoder-only / decoder-only / encoder-decoder / generic). |
| `render.py` | Nested clusters (encoder/decoder ⊇ `N ×` ⊇ sublayers) and Graphviz output: `.png`, `.svg`, `.pdf` or `.dot`. |
| `style.py` | Categories, display names, hyper-parameter details and themes (`paper`, `color`). |

## Options (all keyword arguments of `NNVisualizer`)

- `theme`: `"paper"` | `"color"` | a custom `Theme(...)`
- `collapse`: regex of class names to draw as single boxes, or `f(name, module) -> bool`
- `max_depth`: also collapse every module at this depth
- `collapse_repeats`, `merge_ffn`, `merge_add_norm`, `show_sublayers`
- `fuse_activations`: draw the activation and dropout that follow a Linear layer as a
  line inside its block (`ReLU · Dropout p=0.2`); Feed Forward blocks list theirs too
  (default on)
- `hide_dropout`: hide Dropout layers (shown by default; dropout inside a Feed Forward
  or attention block is drawn as part of that block)
- `hide_inputs`: regex of inputs not to draw (default `mask`)
- `group_pattern`: regex of modules boxed as labelled groups (default encoder/decoder)
- `extra_passes`: your own `f(graph)` rewrites, run after the built-in ones
- render: `direction` (`BT`/`TB`/`LR`), `splines` (`ortho`/`spline`), `show_params`,
  `show_details`, `show_shapes` (tensor shapes on edges), `show_names`, `dpi`, `title`,
  `uniform_width` (every block as wide as the widest, default on), `align_lanes`
  (each data path drawn as a straight, centred column, default on)

Requires PyTorch ≥ 2.1 and the Graphviz `dot` binary (`brew install graphviz`).
