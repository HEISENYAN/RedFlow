import os

import torch
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration

pretrain_model_path = "Qwen/Qwen2.5-VL-3B-Instruct" 
output_model_path = "/mnt/bn/ic-vlm/zhufangqi/code/checkpoints/qwen-vl-resize"

print(f"Loading model and processor: {pretrain_model_path}")

tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
processor = AutoProcessor.from_pretrained(pretrain_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrain_model_path)

print(f"{pretrain_model_path} original vocabulary size: {len(tokenizer)}")

# Add special tokens.
new_tokens = [f"<extra_id_{i}>" for i in range(201)]
num_added_toks = tokenizer.add_tokens(new_tokens, special_tokens=True)
# Resize the embedding and lm_head layers.
model.resize_token_embeddings(len(tokenizer))

# Initialize embeddings for the new tokens.
input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings()

dtype = input_embeddings.weight.dtype
device = input_embeddings.weight.device

original_vocab_limit = len(tokenizer) - num_added_toks
with torch.no_grad():
    existing_weights = input_embeddings.weight[:original_vocab_limit]
    emb_mean = existing_weights.mean(dim=0)
    emb_std = existing_weights.std(dim=0)

    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    for i, token_id in enumerate(new_token_ids):
        token_id = int(token_id)
        if token_id < original_vocab_limit:
            continue
        torch.manual_seed(42 + i)
        init_value = (torch.randn_like(emb_mean) * emb_std * 0.1 + emb_mean).to(device=device, dtype=dtype)
        input_embeddings.weight[token_id] = init_value
        output_embeddings.weight[token_id] = init_value

vocab_size = int(output_embeddings.weight.shape[0])
if vocab_size != len(tokenizer):
    raise RuntimeError(
        f"Embedding rows ({vocab_size}) do not match tokenizer size ({len(tokenizer)})."
    )

model.config.vocab_size = vocab_size
if hasattr(model.config, "text_config") and model.config.text_config is not None:
    model.config.text_config.vocab_size = vocab_size
if getattr(model, "generation_config", None) is not None:
    model.generation_config.vocab_size = vocab_size

print(
    f"Final vocab alignment: tokenizer={len(tokenizer)}, "
    f"input_embeddings={input_embeddings.weight.shape[0]}, "
    f"output_embeddings={output_embeddings.weight.shape[0]}, "
    f"config.vocab_size={model.config.vocab_size}, "
    f"text_config.vocab_size={getattr(getattr(model.config, 'text_config', None), 'vocab_size', 'None')}"
)

print(f"Saving model: {output_model_path}")
os.makedirs(output_model_path, exist_ok=True)

processor.tokenizer = tokenizer
tokenizer.save_pretrained(output_model_path)
processor.save_pretrained(output_model_path)
model.save_pretrained(output_model_path)
