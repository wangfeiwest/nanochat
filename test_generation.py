#!/usr/bin/env python3
"""测试 nanochat 模型生成效果 - 自动选择最新 checkpoint"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
import glob
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# 自动选择最新 checkpoint
checkpoint_dir = '/root/.cache/nanochat/base_checkpoints/d12/'
model_files = glob.glob(f"{checkpoint_dir}/model_*.pt")
model_files.sort(key=lambda x: int(x.split('model_')[1].split('.pt')[0]))
checkpoint_path = model_files[-1]  # 最新 checkpoint
step = int(checkpoint_path.split('model_')[1].split('.pt')[0])

print(f"Loading checkpoint: {checkpoint_path}")
print(f"Step: {step}")

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

# 计算 tokens/params ratio
num_params = sum(p.numel() for p in model.parameters())
tokens_trained = step * 262144  # total batch size * steps
ratio = tokens_trained / num_params

print(f"Parameters: {num_params/1e6:.1f}M")
print(f"Tokens trained: {tokens_trained/1e6:.1f}M")
print(f"Tokens/Params ratio: {ratio:.2f}")

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
print(f"Step {step} 模型生成测试")
print("="*60)

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The most important thing in life is",
    "In the year 2050, humanity",
    "To understand machine learning,",
    "Python is a programming language that",
    "The capital of France is",
]

for prompt in prompts:
    print(f"\n[Prompt]: {prompt}")
    for temp in [0.3, 0.7]:
        output = generate(prompt, max_tokens=30, temperature=temp)
        # 只显示生成的部分
        generated = output[len(prompt):].strip()
        print(f"[Temp {temp}]: {generated[:80]}...")
    print("-"*60)

print("\n" + "="*60)
print("提示: Tokens/Params ratio 需要达到 ~20 才能产生连贯文本")
print(f"当前 ratio {ratio:.2f}，建议继续训练到 step 5000+")
print("="*60)