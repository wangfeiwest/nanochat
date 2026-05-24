#!/bin/bash
# nanochat ROCm training script with float32 (avoid NaN)

export NANOCHAT_DTYPE=float32
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1

cd /mnt/workspace/nanochat

echo "Starting training with NANOCHAT_DTYPE=$NANOCHAT_DTYPE"

python3 -m scripts.base_train \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --num-iterations=20 \
    --run=dummy \
    --core-metric-every=-1 \
    --sample-every=10 \
    --save-every=-1 \
    --eval-every=5 \
    --window-pattern=L

echo "Training completed"