#!/bin/bash
#SBATCH --job-name=gs_train_single
#SBATCH --output=logs/gs_train_frame_%A.out
#SBATCH --error=logs/gs_train_frame_%A.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Usage: sbatch --export=FRAME_IDX=0 train_gs_sbatch_single.sh
# Or: for i in {0..479}; do sbatch --export=FRAME_IDX=$i train_gs_sbatch_single.sh; done

# Configuration
FRAME_IDX=${FRAME_IDX:-0}  # Default to 0 if not provided
DATASET_PATH="/users/wfu16/data/users/wfu16/datasets/2025-11-15/episode_0000"
MODEL_BASE_PATH="/users/wfu16/data/users/wfu16/datasets/2025-11-15/episode_0000"

# Create frame-specific model directory
MODEL_PATH="${MODEL_BASE_PATH}/frame_${FRAME_IDX}"
mkdir -p "${MODEL_PATH}"
mkdir -p logs

# Export environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "========================================="
echo "Training frame ${FRAME_IDX}"
echo "Model path: ${MODEL_PATH}"
echo "========================================="

# Run training for this frame
python gaussian_splatting/train.py \
    -s "${DATASET_PATH}" \
    -m "${MODEL_PATH}" \
    --iterations 14000 \
    --frame_idx ${FRAME_IDX} \
    --test_iterations 1000 5000 7000 10000 14000

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "Frame ${FRAME_IDX} training completed successfully"
else
    echo "ERROR: Frame ${FRAME_IDX} training failed"
    exit 1
fi

