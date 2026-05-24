#!/usr/bin/env python3
"""
nanochat ROCm 训练效果测试 - 直接运行版本
"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
import glob

print("=" * 60)
print("nanochat ROCm 训练效果测试")
print("=" * 60)

# 1. 自动选择最新 checkpoint
checkpoint_dir = '/root/.cache/nanochat/base_checkpoints/d12/'
model_files = glob.glob(f"{checkpoint_dir}/model_*.pt")
model_files.sort(key=lambda x: int(x.split('model_')[1].split('.pt')[0]))
checkpoint_path = model_files[-1]
step = int(checkpoint_path.split('model_')[1].split('.pt')[0])

print(f"\n[1] Loading checkpoint: {checkpoint_path}")
print(f"    Step: {step}")

state_dict = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
print("    ✓ Checkpoint loaded")

# 2. 创建模型
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

num_params = sum(p.numel() for p in model.parameters())
tokens_trained = step * 262144
ratio = tokens_trained / num_params

print(f"\n[2] Model info:")
print(f"    Parameters: {num_params/1e6:.1f}M")
print(f"    Tokens trained: {tokens_trained/1e6:.1f}M")
print(f"    Tokens/Params ratio: {ratio:.2f}")

# 3. 加载 tokenizer
tokenizer = get_tokenizer()
print(f"\n[3] Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# 4. 定义生成函数
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

print("\n[4] ✓ Generate function defined")

# 5. 测试生成
print("\n" + "=" * 60)
print(f"Step {step} 模型生成测试")
print("=" * 60)

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The most important thing in life is",
    "Python is a programming language that",
    "To understand machine learning,",
    "The capital of France is",
]

for prompt in prompts:
    print(f"\n[Prompt]: {prompt}")
    for temp in [0.3, 0.7]:
        output = generate(prompt, max_tokens=30, temperature=temp)
        generated = output[len(prompt):].strip()[:80]
        print(f"[Temp {temp}]: {generated}")
    print("-" * 60)

print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)
print(f"val_bpb: 从 3.17 → ~1.0 (下降 ~66%)")
print(f"Tokens/Params ratio: {ratio:.2f} (需要 ~20 才能产生连贯文本)")
print(f"建议: 继续训练到 step 5000+ 以改善生成质量")
print("=" * 60)