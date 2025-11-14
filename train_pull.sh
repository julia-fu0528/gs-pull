export PYTHONNOUSERSITE=1

unset PYTHONPATH
python train.py \
    -s /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -c /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --output /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --dataset_name brics --eval True