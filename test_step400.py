#!/usr/bin/env python3
"""测试 nanochat 模型生成效果"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# 加载 step 400 checkpoint
checkpoint_path = '/root/.cache/nanochat/base_checkpoints/d12/model_000400.pt'
print(f"Loading checkpoint: {checkpoint_path}")

state_dict = torch.load(checkpoint_path, map_location='cuda', weights_only=True)

config = GPTConfig(
    sequence_len=2048,
    vocab_size=32768,
    n_layer=12,
    n_head=6,
    n_kv_head=6,
    n_embd=768,
    window_pattern='L'
)

model = GPT(config)
model.load_state_dict(state_dict)
model = model.to('cuda').to(torch.float32)
model.eval()

tokenizer = get_tokenizer()
print(f"Model loaded! Step 400 checkpoint")

def generate(prompt, max_tokens=50, temperature=0.7, top_k=40):
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor([encoded], device='cuda')
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens[:, -2048:] if tokens.shape[1] > 2048 else tokens)
            logits = logits[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())

print("\n" + "="*60)
print("Step 400 模型生成测试")
print("="*60)

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The most important thing in life is",
    "In the year 2050, humanity",
    "To understand machine learning,",
]

for prompt in prompts:
    print(f"\n[Prompt]: {prompt}")
    for temp in [0.3, 0.7]:
        output = generate(prompt, max_tokens=30, temperature=temp)
        print(f"[Temp {temp}]: {output[len(prompt):]}")
    print("-"*60)