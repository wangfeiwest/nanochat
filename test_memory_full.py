#!/usr/bin/env python3
"""测试不同模型配置的显存使用"""
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
import gc

def test_config(depth, seq_len, batch_size):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 根据深度计算模型参数
    # depth=12: n_head=6, n_embd=768
    # depth=24: n_head=12, n_embd=1536
    # depth=36: n_head=18, n_embd=2304
    n_embd = 768 * (depth // 12)
    n_head = 6 * (depth // 12)
    
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=32768,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern='L'
    )
    
    try:
        model = GPT(config).to('cuda').to(torch.float32)
        num_params = sum(p.numel() for p in model.parameters())
        
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
        
        del model, x, y, loss, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return peak_mem, num_params, True
    except RuntimeError as e:
        gc.collect()
        torch.cuda.empty_cache()
        return 0, 0, False

print("="*70)
print("显存使用测试 - 寻找最优配置")
print("="*70)
print(f"GPU 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# 测试不同配置
configs = [
    # (depth, seq_len, batch_size)
    (12, 2048, 64),   # 当前配置
    (12, 4096, 32),   # 更长序列
    (12, 4096, 64),   # 更长序列 + 更大 batch
    (24, 2048, 16),   # 更深模型
    (24, 2048, 32),   # 更深模型 + 更大 batch
    (24, 4096, 16),   # 更深 + 更长序列
    (36, 2048, 8),    # 更更深
    (48, 2048, 4),    # depth=48
]

print(f"{'Config':<25} {'Params':>15} {'Peak Mem':>12} {'Status':>10}")
print("-"*70)

for depth, seq_len, batch_size in configs:
    peak, params, success = test_config(depth, seq_len, batch_size)
    config_str = f"d{depth}_seq{seq_len}_bs{batch_size}"
    params_str = f"{params/1e6:.1f}M" if params > 0 else "-"
    mem_str = f"{peak:.1f}GB" if peak > 0 else "-"
    status = "OK" if success else "OOM"
    
    print(f"{config_str:<25} {params_str:>15} {mem_str:>12} {status:>10}")
    
    if success and peak > 150:
        print(f"\n*** 推荐配置: depth={depth}, seq_len={seq_len}, batch_size={batch_size}")
        print(f"*** 预计参数: {params/1e6:.1f}M, 显存: {peak:.1f}GB")

print()
print("="*70)