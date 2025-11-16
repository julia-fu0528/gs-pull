#!/bin/bash
# Script to submit all 480 frames for training
# Option 1: Submit as array job (10 frames per GPU, 48 GPUs total)
# Option 2: Submit individual jobs (1 frame per GPU, 480 GPUs total)

TOTAL_FRAMES=27
FRAMES_PER_GPU=1
NUM_GPUS=$(( (TOTAL_FRAMES + FRAMES_PER_GPU - 1) / FRAMES_PER_GPU ))  # Ceiling division

echo "Total frames: ${TOTAL_FRAMES}"
echo "Frames per GPU: ${FRAMES_PER_GPU}"
echo "Number of GPUs needed: ${NUM_GPUS}"

# Option 1: Submit as array job (recommended - more efficient)
echo ""
echo "Option 1: Submitting as array job (${NUM_GPUS} jobs, ${FRAMES_PER_GPU} frames per job)..."
sbatch train_gs_sbatch.sh

# Option 2: Submit individual jobs (uncomment to use)
# echo ""
# echo "Option 2: Submitting individual jobs (${TOTAL_FRAMES} jobs, 1 frame per job)..."
# for i in $(seq 0 $((TOTAL_FRAMES - 1))); do
#     sbatch --export=FRAME_IDX=$i train_gs_sbatch_single.sh
#     sleep 0.1  # Small delay to avoid overwhelming the scheduler
# done

echo ""
echo "Jobs submitted. Check status with: squeue -u $USER"

