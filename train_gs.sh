export PYTHONNOUSERSITE=1

unset PYTHONPATH
python gaussian_splatting/train.py \
    -s /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -m /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --iterations 16000 --frame_idx 0 \
    --test_iterations 1000 5000 7000 10000 14000 16000\