#!/bin/bash
# nanochat ROCm training - use float32 to avoid NaN

export NANOCHAT_DTYPE=float32
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1

cd /mnt/workspace/nanochat

# Train depth=12 model (GPT-1 scale)
# Note: float32 is slower but stable on ROCm

python3 -m scripts.base_train \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=262144 \
    --num-iterations=500 \
    --run=dummy \
    --core-metric-every=-1 \
    --sample-every=100 \
    --save-every=100 \
    --eval-every=50 \
    --window-pattern=L \
    2>&1 | tee train.log

echo "Training completed. Check train.log for results."