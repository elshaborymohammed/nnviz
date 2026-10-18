"""Best-effort example inputs so a raw model can be drawn without arguments."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def make_dummy_input(model: nn.Module, seq_len: int = 8, batch: int = 1) -> Any:
    """Guess a valid input for ``model``. Pass your own input when this fails."""
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    config = getattr(model, "config", None)

    if config is not None:  # HuggingFace models
        if hasattr(config, "num_channels") and hasattr(config, "image_size"):
            size = config.image_size if isinstance(config.image_size, int) else config.image_size[0]
            return {"pixel_values": torch.randn(batch, config.num_channels, size, size, device=device)}
        vocab = getattr(config, "vocab_size", 100) or 100
        ids = torch.randint(0, min(vocab, 100), (batch, seq_len), device=device)
        inputs = {"input_ids": ids}
        if getattr(config, "is_encoder_decoder", False):
            inputs["decoder_input_ids"] = ids.clone()
        if "use_cache" in config.to_dict():
            inputs["use_cache"] = False
        return inputs

    if isinstance(model, nn.Transformer):
        bf = model.encoder.layers[0].self_attn.batch_first
        shape = (batch, seq_len, model.d_model) if bf else (seq_len, batch, model.d_model)
        return {"src": torch.randn(shape, device=device),
                "tgt": torch.randn(shape, device=device),
                "tgt_mask": nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)}

    first = next((m for m in model.modules() if not any(True for _ in m.children())
                  or isinstance(m, nn.MultiheadAttention)), None)
    if isinstance(first, nn.Embedding):
        return torch.randint(0, min(first.num_embeddings, 100), (batch, seq_len), device=device)
    if isinstance(first, nn.Linear):
        return torch.randn(batch, first.in_features, device=device)
    if isinstance(first, nn.Conv1d):
        return torch.randn(batch, first.in_channels, 64, device=device)
    if isinstance(first, nn.Conv2d):
        return torch.randn(batch, first.in_channels, 64, 64, device=device)
    if isinstance(first, nn.Conv3d):
        return torch.randn(batch, first.in_channels, 16, 32, 32, device=device)
    if isinstance(first, (nn.LSTM, nn.GRU, nn.RNN)):
        shape = (batch, seq_len, first.input_size) if first.batch_first else (seq_len, batch, first.input_size)
        return torch.randn(shape, device=device)
    if isinstance(first, (nn.LayerNorm,)):
        return torch.randn(batch, seq_len, *first.normalized_shape, device=device)
    if isinstance(first, nn.MultiheadAttention):
        shape = (batch, seq_len, first.embed_dim) if first.batch_first else (seq_len, batch, first.embed_dim)
        return torch.randn(shape, device=device)
    raise ValueError(f"Cannot guess an input for {type(model).__name__}; "
                     "pass `input_data=` (tensor, tuple of args, or dict of kwargs).")
