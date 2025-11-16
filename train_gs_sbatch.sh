#!/bin/bash
#SBATCH --job-name=gs_train
#SBATCH --output=logs/gs_train_%A_%a.out
#SBATCH --error=logs/gs_train_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-26%1  # 27 jobs (0-26), run max 1 at a time, each handles 1 frame

# Configuration
FRAMES_PER_GPU=1
TOTAL_FRAMES=27
BASE_FRAME_IDX=$((SLURM_ARRAY_TASK_ID * FRAMES_PER_GPU))
DATASET_PATH="/users/wfu16/data/users/wfu16/datasets/2025-11-15/episode_0000"
MODEL_BASE_PATH="/users/wfu16/data/users/wfu16/datasets/2025-11-15/episode_0000"

# Create logs directory
mkdir -p logs

# Export environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Loop through frames for this GPU
for i in $(seq 0 $((FRAMES_PER_GPU - 1))); do
    FRAME_IDX=$((BASE_FRAME_IDX + i))
    
    # Skip if we've exceeded total frames
    if [ $FRAME_IDX -ge $TOTAL_FRAMES ]; then
        break
    fi
    
    # Create frame-specific model directory
    MODEL_PATH="${MODEL_BASE_PATH}/frame_${FRAME_IDX}"
    mkdir -p "${MODEL_PATH}"
    
    echo "========================================="
    echo "Processing frame ${FRAME_IDX} on GPU ${SLURM_ARRAY_TASK_ID}"
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
    
    echo ""
done

echo "All frames for GPU ${SLURM_ARRAY_TASK_ID} completed"

