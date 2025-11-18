export PYTHONNOUSERSITE=1

unset PYTHONPATH
python extract_mesh.py \
    -s /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -g /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -o /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --dataset_name brics \
    # --cloth \
    # --thickness 0.001 