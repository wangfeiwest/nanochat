#!/usr/bin/env python3
"""测试不同 batch size 的显存使用情况"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
import gc

torch.cuda.reset_peak_memory_stats()

config = GPTConfig(
    sequence_len=2048,
    vocab_size=32768,
    n_layer=12,
    n_head=6,
    n_kv_head=6,
    n_embd=768,
    window_pattern='L'
)

def test_batch_size(batch_size, seq_len=2048):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = GPT(config).to('cuda').to(torch.float32)
    
    # 模拟训练一个 step
    x = torch.randint(0, 32768, (batch_size, seq_len), device='cuda')
    y = torch.randint(0, 32768, (batch_size, seq_len), device='cuda')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    result = model(x, targets=y)
    if isinstance(result, tuple):
        logits, loss = result
    else:
        loss = result
    loss.backward()
    optimizer.step()
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    current_mem = torch.cuda.memory_allocated() / 1024**3
    
    del model, x, y, loss, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return peak_mem, current_mem

print("="*60)
print("显存使用测试 (depth=12, seq_len=2048)")
print("="*60)
print(f"GPU 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

batch_sizes = [4, 8, 16, 32, 64, 128, 256]

for bs in batch_sizes:
    try:
        peak, current = test_batch_size(bs)
        print(f"batch_size={bs:3d}: Peak={peak:.2f}GB, Current={current:.2f}GB")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"batch_size={bs:3d}: OOM (显存不足)")
            break
        else:
            print(f"batch_size={bs:3d}: Error - {e}")
            break

print()
print("建议配置:")
print("="*60)