export PYTHONNOUSERSITE=1

unset PYTHONPATH
python evaluation/clean_eval_dtu_mesh.py \
    --datadir /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --expdir /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --scan 0000