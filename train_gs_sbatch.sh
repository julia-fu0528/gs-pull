#!/bin/bash
#SBATCH --job-name=gs_train
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/gs_train_%A_%a.out
#SBATCH --error=logs/gs_train_%A_%a.err
#SBATCH --array=0-47%10  # 48 jobs (0-47), run max 10 at a time, each handles 10 frames

# Configuration
FRAMES_PER_GPU=10
TOTAL_FRAMES=480
BASE_FRAME_IDX=$((SLURM_ARRAY_TASK_ID * FRAMES_PER_GPU))
DATASET_PATH="/users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000"
MODEL_BASE_PATH="/users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000"

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
    
    # Use a unique port for each job to avoid conflicts
    # Base port 6009 + array_task_id * 100 + frame_offset to ensure uniqueness
    PORT=$((6009 + SLURM_ARRAY_TASK_ID * 100 + i))
    
    echo "========================================="
    echo "Processing frame ${FRAME_IDX} on GPU ${SLURM_ARRAY_TASK_ID}"
    echo "Model path: ${MODEL_PATH}"
    echo "Port: ${PORT}"
    echo "========================================="
    
    # Run training for this frame
    python gaussian_splatting/train.py \
        -s "${DATASET_PATH}" \
        -m "${MODEL_PATH}" \
        --iterations 16000 \
        --frame_idx ${FRAME_IDX} \
        --port ${PORT} \
        --test_iterations 1000 5000 7000 10000 14000 16000
    
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

