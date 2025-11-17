
REMOVE_CAMS="brics-odroid-003_cam0,brics-odroid-003_cam1,\
brics-odroid-004_cam0,\
brics-odroid-008_cam0,brics-odroid-008_cam1,\
brics-odroid-009_cam0,\
brics-odroid-013_cam0,brics-odroid-013_cam1,\
brics-odroid-014_cam0,\
brics-odroid-018_cam0,brics-odroid-018_cam1,\
brics-odroid-019_cam0,\
brics-odroid-027_cam0,brics-odroid-027_cam1,\
brics-odroid-028_cam0,"

export PYTHONNOUSERSITE=1

unset PYTHONPATH
python train.py \
    -s /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -c /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    -i 36000 \
    --output /users/wfu16/data/users/wfu16/datasets/2025-10-29-cloth/episode_0000 \
    --dataset_name brics --eval True \
    # --remove_cams $REMOVE_CAMS
