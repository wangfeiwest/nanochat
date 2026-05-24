---
name: nanochat-rocm-training
description: Train nanochat GPT models on AMD ROCm GPUs (MI300X, MI250X). Solve NaN loss issues and optimize VRAM usage.
tags:
  - rocm
  - amd
  - training
  - gpt
  - nanochat
  - optimizer
version: "1.0"
---

# nanochat ROCm Training

Train karpathy/nanochat on AMD ROCm GPUs with proper optimizer and VRAM optimization.

## Prerequisites

- AMD ROCm GPU (MI300X, MI250X, etc.)
- ROCm 6.x installed
- PyTorch ROCm version
- Python 3.10+

## Key Modifications for ROCm

### 1. Replace Muon Optimizer with AdamW

Muon's Newton-Schulz iteration causes NaN loss on ROCm. Replace in `nanochat/gpt.py`:

```python
# Find setup_optimizer method (~line 402)
# Replace Muon param groups with AdamW:

for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='adamw', params=group_params, lr=matrix_lr,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
    ))

# Replace MuonAdamW with standard AdamW:
import torch.optim as optim
optimizer = optim.AdamW(param_groups)
```

### 2. Use Float32 dtype

Set `NANOCHAT_DTYPE=float32` to prevent dtype issues:

```bash
export NANOCHAT_DTYPE=float32
```

### 3. Disable Sample Generation (optional)

Sample generation has dtype mismatch with kv_cache. Use `--sample-every=-1` during training.

## VRAM Optimization

MI300X has 192GB HBM3. Optimize VRAM usage by increasing batch size:

```bash
# Test VRAM usage for different batch sizes
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from nanochat.gpt import GPT, GPTConfig

config = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=12, n_head=6, n_kv_head=6, n_embd=768, window_pattern='L')
model = GPT(config).to('cuda').to(torch.float32)

for bs in [4, 16, 32, 64, 128]:
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 32768, (bs, 2048), device='cuda')
    result = model(x)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f'batch_size={bs}: Peak={peak:.1f}GB')
"
```

**Recommended**: `device-batch-size=64` achieves ~89% VRAM utilization (171GB).

## Training Command

```bash
cd /path/to/nanochat

export NANOCHAT_DTYPE=float32 WANDB_MODE=disabled NANOCHAT_NO_COMPILE=1

python3 -u -m scripts.base_train \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=64 \
    --total-batch-size=262144 \
    --num-iterations=1700 \
    --run=rocm_training \
    --sample-every=-1 \
    --save-every=200 \
    --eval-every=100 \
    --window-pattern L
```

## Resume Training

```bash
# Resume from checkpoint
python3 -u -m scripts.base_train \
    --depth=12 \
    --resume-from-step=200 \
    --num-iterations=1700 \
    ...
```

## Monitoring

```bash
# ROCm GPU monitoring
rocm-smi --showuse --showpower --showmeminfo vram
```

## Common Issues

| Issue | Solution |
|-------|----------|
| NaN loss at step 2 | Replace Muon with AdamW |
| dtype mismatch in sample | Use `--sample-every=-1` |
| Low VRAM usage | Increase `device-batch-size` |
| wandb login error | Set `WANDB_MODE=disabled` |

## Performance Metrics

| Config | Token/sec | VRAM | MFU |
|--------|-----------|------|-----|
| bs=4 | 31,867 | 14GB | 2.16% |
| bs=64 | 35,679 | 171GB | 2.42% |

## NVIDIA vs ROCm Differences

| Aspect | NVIDIA | ROCm |
|--------|--------|------|
| Flash Attention | FA3 supported | SDPA fallback |
| Muon Optimizer | Works | NaN (use AdamW) |
| dtype | bf16 OK | float32 recommended |
| GPU Monitor | nvidia-smi | rocm-smi |
| MFU | 40-50% | 2-3% |

## Files Modified

1. `nanochat/gpt.py` - setup_optimizer method
2. `scripts/base_train.py` - NANOCHAT_NO_COMPILE env var (optional)

## References

- Repository: https://github.com/karpathy/nanochat
- Full guide: See `ROCm_DEPLOYMENT_GUIDE.md` in nanochat directory