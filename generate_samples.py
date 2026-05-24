#!/usr/bin/env python3
"""Generate text samples from trained nanochat model on ROCm"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
import json

# Load checkpoint
checkpoint_path = '/root/.cache/nanochat/base_checkpoints/d12/model_000200.pt'
print(f"Loading checkpoint: {checkpoint_path}")

state_dict = torch.load(checkpoint_path, map_location='cuda', weights_only=True)

# Get model config from metadata
meta_path = '/root/.cache/nanochat/base_checkpoints/d12/meta_000200.json'
with open(meta_path) as f:
    meta = json.load(f)

config = GPTConfig(
    sequence_len=2048,
    vocab_size=32768,
    n_layer=12,
    n_head=6,
    n_kv_head=6,
    n_embd=768,
    window_pattern='L'
)

print("Creating model...")
model = GPT(config)
model.load_state_dict(state_dict)
model = model.to('cuda')
model = model.to(torch.float32)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = get_tokenizer()

print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate samples
def generate(prompt_tokens, max_tokens=50, temperature=1.0, top_k=50):
    model.eval()
    tokens = prompt_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits for last token
            logits = model(tokens[:, -2048:] if tokens.shape[1] > 2048 else tokens)
            logits = logits[:, -1, :]  # last position
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokens

# Test prompts
prompts = [
    "The future of artificial intelligence",
    "Once upon a time in a distant galaxy",
    "The most important thing in life is",
    "In the year 2050, humanity",
]

print("\n" + "="*60)
print("GENERATED SAMPLES")
print("="*60)

temperatures = [0.3, 0.7, 1.0]
prompt = "The future of artificial intelligence is"

for temp in temperatures:
    print(f"\n[Temperature: {temp}]")
    print(f"[Prompt]: {prompt}")
    encoded = tokenizer.encode(prompt)
    prompt_tokens = torch.tensor([[encoded]], device='cuda').squeeze(1)
    if prompt_tokens.dim() == 3:
        prompt_tokens = prompt_tokens.squeeze(0)
    generated = generate(prompt_tokens, max_tokens=50, temperature=temp, top_k=40)
    text = tokenizer.decode(generated[0].tolist())
    print(f"[Generated]: {text}")
    print("-"*40)

# Longer generation
print("\n" + "="*60)
print("LONGER GENERATION (100 tokens)")
print("="*60)
prompt = "In a world where technology"
print(f"[Prompt]: {prompt}")
encoded = tokenizer.encode(prompt)
prompt_tokens = torch.tensor([[encoded]], device='cuda').squeeze(1)
if prompt_tokens.dim() == 3:
    prompt_tokens = prompt_tokens.squeeze(0)
generated = generate(prompt_tokens, max_tokens=100, temperature=0.7, top_k=50)
text = tokenizer.decode(generated[0].tolist())
print(f"[Generated]: {text}")

print("\nDone!")