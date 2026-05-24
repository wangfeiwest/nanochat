# nanochat 在 AMD ROCm 环境部署教程

## 概述

本教程记录了在 AMD MI300X GPU (ROCm 6.10.5) 环境部署和训练 karpathy/nanochat 的完整过程，重点指出与 NVIDIA CUDA 环境的差异和需要修改的部分。

---

## 环境信息

### 本测试环境
- **GPU**: AMD MI300X (gfx942), 192GB HBM3
- **ROCm**: 6.10.5
- **PyTorch**: 2.10.0 (ROCm 7.2)
- **Python**: 3.12.13
- **Peak FLOPS**: 1.31e+15 (BF16)

### NVIDIA 环境对比
- NVIDIA 使用 CUDA, Flash Attention 3 官方支持
- AMD ROCm 需要使用 PyTorch SDPA fallback (FA3 不支持)

---

## 部署步骤

### 1. 克隆仓库

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

### 2. 安装依赖

```bash
pip install wandb rustbpe huggingface_hub
```

**AMD 环境注意事项**:
- PyTorch ROCm 版本需要单独安装（通常预装在 ROCm 环境）
- 确认 PyTorch 能识别 GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### 3. 下载数据集

**中国大陆网络环境**需要使用 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 下载 climbmix 数据集 (约 14GB, 155 shards)
python3 -c "
from huggingface_hub import hf_hub_download
import os

repo_id = 'karpathy/climbmix-400b-shuffle'
local_dir = '/root/.cache/nanochat/base_data_climbmix'

for i in range(155):
    filename = f'shard_{i:05d}.parquet'
    print(f'Downloading {filename}...')
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='dataset',
        local_dir=local_dir
    )
print('Done!')
"
```

**与 NVIDIA 环境差异**: 无差异，但网络环境可能需要镜像

### 4. 训练 Tokenizer

```bash
python3 -c "
from nanochat.tokenizer import train_tokenizer
train_tokenizer()
"
```

Tokenizer 保存到 `~/.cache/nanochat/tokenizer/tokenizer.pkl`

---

## 关键修改：解决 ROCm 兼容性问题

### 问题 1: Muon Optimizer 数值不稳定

**症状**: 训练在 step 2 出现 NaN loss

**原因**: Muon optimizer 的 Newton-Schulz 迭代在 ROCm 上数值不稳定

**解决方案**: 将 Muon optimizer 替换为标准 AdamW

修改文件: `nanochat/gpt.py` (约 line 402-416)

```python
# 原代码 (Muon optimizer)
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)

# 修改后 (使用 AdamW)
        # AdamW groups for ALL params (modified for ROCm compatibility)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='adamw', params=group_params, lr=matrix_lr,
                betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
            ))

        # Use standard AdamW for ROCm
        import torch.optim as optim
        optimizer = optim.AdamW(param_groups)
```

### 问题 2: Flash Attention 3 不支持

**症状**: 警告信息 "Flash Attention 3 not available, using PyTorch SDPA fallback"

**原因**: AMD ROCm 暂不支持 Flash Attention 3

**影响**: 
- 训练效率降低 (~2.16% MFU vs NVIDIA 可能更高)
- 但功能正常，可以训练

**无需修改**: nanochat 自动 fallback 到 PyTorch SDPA

### 问题 3: GPU Peak FLOPS 显示异常

**症状**: GPU 名称显示为空，Peak FLOPS 显示异常值

**解决方案**: 修改 `nanochat/common.py` 添加 ROCm GPU 检测

```python
# 在 get_peak_flops 函数中添加 ROCm 支持
def get_peak_flops():
    device_type = autodetect_device_type()
    if device_type == 'cuda':
        # ROCm 环境
        import subprocess
        try:
            result = subprocess.run(['rocm-smi', '--showgpuname'], capture_output=True, text=True)
            gcn_arch = result.stdout.strip()
            
            # MI300X (gfx942) = 1.31e15 BF16 FLOPS
            if 'gfx942' in gcn_arch or 'MI300' in gcn_arch:
                return 1.31e15
            # 其他 AMD GPU...
        except:
            pass
    # ... 其他逻辑
```

### 问题 4: Sample 生成的 dtype 不一致

**症状**: 生成样本时 "Expected query, key, and value to have the same dtype"

**原因**: kv_cache 使用 bf16 但模型用 float32

**解决方案**: 禁用 sample 生成或修改 flash_attention.py

训练时使用 `--sample-every=-1` 禁用 sample

---

## 训练命令

### 基础训练命令

```bash
cd /mnt/workspace/nanochat

export NANOCHAT_DTYPE=float32 WANDB_MODE=disabled NANOCHAT_NO_COMPILE=1

# 高显存利用率配置 (推荐)
python3 -u -m scripts.base_train \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=64 \
    --total-batch-size=262144 \
    --num-iterations=5000 \
    --run=rocm_d12 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=500 \
    --eval-every=250 \
    --window-pattern L
```

### 参数说明

| 参数 | 说明 | ROCm 注意事项 |
|------|------|--------------|
| `NANOCHAT_DTYPE=float32` | 使用 float32 防止 NaN | NVIDIA 可用 bf16 |
| `NANOCHAT_NO_COMPILE=1` | 禁用 torch.compile | 加速启动测试 |
| `WANDB_MODE=disabled` | 禁用 wandb | 需要登录否则报错 |
| `--sample-every=-1` | 禁用 sample 生成 | 避免 dtype 问题 |

---

## 性能对比

### 本环境 (AMD MI300X)

#### 低显存配置 (device-batch-size=4)

| 指标 | 值 |
|------|-----|
| Token/sec | ~31,867 |
| BF16 MFU | 2.16% |
| 每步耗时 | 8.23 秒 |
| 显存使用 | ~14 GB (7%) |

#### 高显存配置 (device-batch-size=64) - 推荐

| 指标 | 值 |
|------|-----|
| Token/sec | ~35,679 |
| BF16 MFU | 2.42% |
| 每步耗时 | 7.35 秒 |
| 显存峰值 | ~171 GB (89%) |

### 显存优化策略

MI300X 有 192GB HBM3 显存，通过增加 `device-batch-size` 可大幅提升利用率：

```bash
# 测试不同 batch size 的显存使用
python3 test_memory.py

# 结果示例:
# batch_size=4:   10.75 GB
# batch_size=64:  154.65 GB  ← 推荐
# batch_size=128: OOM
```

**显存组成分析**:
- 模型权重: ~1.1 GB
- Optimizer state (AdamW): ~2.2 GB (模型的 2x)
- 激活值 (backward峰值): ~150 GB
- 总峰值: ~154 GB

**关键**: 峰值显存在 backward 时达到，rocm-smi 显示的是当前显存而非峰值。训练脚本会报告 `Peak memory usage`。

### NVIDIA 环境预期

| 指标 | 预期值 |
|------|--------|
| Token/sec | 更高 (FA3 支持) |
| BF16 MFU | ~40-50% |
| 每步耗时 | 更低 |

**MFU 差异原因**:
- Flash Attention 3 在 NVIDIA 上大幅提升效率
- ROCm 使用 SDPA fallback，效率较低

---

## 监控 GPU 使用

```bash
# AMD ROCm
rocm-smi --showuse --showpower --showmeminfo vram

# NVIDIA CUDA
nvidia-smi
```

---

## 文本生成

训练完成后，使用以下脚本生成文本：

```python
#!/usr/bin/env python3
import os
os.environ['NANOCHAT_DTYPE'] = 'float32'

import torch
import sys
sys.path.insert(0, '/mnt/workspace/nanochat')

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# 加载模型
checkpoint_path = '/root/.cache/nanochat/base_checkpoints/d12/model_000200.pt'
state_dict = torch.load(checkpoint_path, map_location='cuda', weights_only=True)

config = GPTConfig(
    sequence_len=2048, vocab_size=32768,
    n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
    window_pattern='L'
)

model = GPT(config)
model.load_state_dict(state_dict)
model = model.to('cuda').to(torch.float32)
tokenizer = get_tokenizer()

# 生成函数
def generate(prompt_tokens, max_tokens=50, temperature=0.7, top_k=40):
    model.eval()
    tokens = prompt_tokens.clone()
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
    return tokens

# 使用示例
prompt = "The future of AI is"
encoded = tokenizer.encode(prompt)
prompt_tokens = torch.tensor([[encoded]], device='cuda').squeeze(1)
if prompt_tokens.dim() == 3:
    prompt_tokens = prompt_tokens.squeeze(0)

generated = generate(prompt_tokens, max_tokens=50, temperature=0.7)
print(tokenizer.decode(generated[0].tolist()))
```

---

## Checkpoint 位置

```
~/.cache/nanochat/
├── base_data_climbmix/      # 数据集
├── tokenizer/               # Tokenizer
│   └── tokenizer.pkl
└── base_checkpoints/
    └── d12/                 # depth=12 模型
        ├── model_000200.pt  # 模型权重
        ├── meta_000200.json # 元数据
        └── optim_000200_rank0.pt  # Optimizer 状态
```

---

## 总结：ROCm vs CUDA 关键差异

| 差异项 | NVIDIA CUDA | AMD ROCm |
|--------|-------------|----------|
| Flash Attention | FA3 官方支持 | SDPA fallback |
| Optimizer | Muon 正常工作 | 需替换为 AdamW |
| GPU 监控 | nvidia-smi | rocm-smi |
| Peak FLOPS | 自动检测 | 需手动配置 |
| dtype | bf16 正常 | 建议 float32 |
| MFU 效率 | ~40-50% | ~2% |

---

## 附录：完整修改补丁

### gpt.py optimizer 修改

```diff
--- a/nanochat/gpt.py
+++ b/nanochat/gpt.py
@@ -399,16 +399,17 @@
             dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
             dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
         ]
-        # Muon groups (matrix params, grouped by shape for stacking)
+        # AdamW groups for ALL params (modified for ROCm compatibility)
         for shape in sorted({p.shape for p in matrix_params}):
             group_params = [p for p in matrix_params if p.shape == shape]
             param_groups.append(dict(
-                kind='muon', params=group_params, lr=matrix_lr,
-                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
+                kind='adamw', params=group_params, lr=matrix_lr,
+                betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
             ))

-        Factory = DistMuonAdamW if ddp else MuonAdamW
-        optimizer = Factory(param_groups)
+        import torch.optim as optim
+        optimizer = optim.AdamW(param_groups)
         for group in optimizer.param_groups:
             group["initial_lr"] = group["lr"]
         return optimizer
```

### base_train.py 添加 torch.compile 控制

```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -243,7 +243,10 @@
 # Compile the model

 orig_model = model
-model = torch.compile(model, dynamic=False)
+if os.environ.get('NANOCHAT_NO_COMPILE', '0') == '1':
+    print0("Skipping torch.compile (NANOCHAT_NO_COMPILE=1)")
+else:
+    model = torch.compile(model, dynamic=False)
```

---

## 参考资料

- nanochat 仓库: https://github.com/karpathy/nanochat
- ROCm 文档: https://rocm.docs.amd.com
- PyTorch ROCm: https://pytorch.org/get-started/locally/#rocm-installation