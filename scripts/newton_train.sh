#!/bin/bash
#SBATCH --job-name=paramgen
#SBATCH --output=logs/paramgen_train_%j.out
#SBATCH --error=logs/paramgen_train_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

###############################################################################
# Stage 3 — Train Parameter Generator (HyperNetwork)
#
# Prerequisites:
#   1. Stage 1 (Data Factory) completed
#   2. Stage 2 (LoRA Factory) completed
#   3. collect_adapters.py has been run to create paramgen_dataset.pt
#
# Usage:
#   sbatch tools/param_generator/newton_train_paramgen.sh
###############################################################################

set -euo pipefail

PROJECT_DIR="$HOME/VGen_transfer"
CONDA_ENV="dreamvideo"

cd "$PROJECT_DIR"
mkdir -p logs

export OPENBLAS_NUM_THREADS=4
module load cuda/11.8 2>/dev/null || true
export PATH="$HOME/.conda/envs/$CONDA_ENV/bin:$PATH"

echo "=== Stage 3: Train Parameter Generator ==="
echo "Start: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname)"

DATASET="workspace/paramgen_dataset/paramgen_dataset.pt"
OUT_DIR="workspace/paramgen_training"

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run collect_adapters.py first!"
    exit 1
fi

python3 tools/param_generator/train_paramgen.py \
    --dataset "$DATASET" \
    --out-dir "$OUT_DIR" \
    --epochs 500 \
    --batch-size 8 \
    --lr 1e-4 \
    --hidden-dim 2048 \
    --backbone-layers 4 \
    --dropout 0.1 \
    --warmup-steps 100 \
    --loss-weight-mse 1.0 \
    --loss-weight-cosine 0.5 \
    --loss-weight-reg 0.01

echo "=== Training Complete ==="
echo "End: $(date)"
echo "Best model: $OUT_DIR/paramgen_best.pt"
