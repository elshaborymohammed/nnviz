"""HuggingFace architectures built from small configs (no downloads)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import (BertConfig, BertModel, GPT2Config, GPT2LMHeadModel, T5Config,
                          T5ForConditionalGeneration, LlamaConfig, LlamaForCausalLM, ViTConfig, ViTModel)
from nnviz import NNVisualizer
OUT = Path(__file__).resolve().parents[1] / "out"
models = {
 "hf_bert": BertModel(BertConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512)),
 "hf_gpt2": GPT2LMHeadModel(GPT2Config(n_embd=128, n_layer=4, n_head=4)),
 "hf_t5": T5ForConditionalGeneration(T5Config(d_model=128, d_ff=512, num_layers=3, num_heads=4, d_kv=32)),
 "hf_llama": LlamaForCausalLM(LlamaConfig(hidden_size=128, intermediate_size=344, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)),
 "hf_vit": ViTModel(ViTConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512, image_size=64, patch_size=16)),
}
for name, m in models.items():
    try:
        v = NNVisualizer(m)
        v.draw(OUT / f"{name}.png")
        print(v.summary())
    except Exception:
        print("FAILED", name); traceback.print_exc()
