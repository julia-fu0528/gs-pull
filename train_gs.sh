export PYTHONNOUSERSITE=1

unset PYTHONPATH
python gaussian_splatting/train.py \
    -s /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -m /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --iterations 20000 --start_frame 0 --end_frame 480 --num_frames 480 \
    --test_iterations 1000 5000 7000 10000 15000 20000 \