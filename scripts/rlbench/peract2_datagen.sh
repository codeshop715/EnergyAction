DATA_PATH=peract2_raw/
ZARR_PATH=zarr_datasets/peract2/

# This can be slow: download and unsquash all data
python scripts/rlbench/download_peract2.py --root ${DATA_PATH}

# Download the test seeds
CURR_DIR=$(pwd)
cd ${DATA_PATH}
wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
unzip peract2_test.zip
rm peract2_test.zip
cd "$CURR_DIR"

# Now we just need to package them
python data_processing/peract_to_zarr.py \
    --root ${DATA_PATH} \
    --tgt ${ZARR_PATH}
# You can safely delete the train and val raw data now, not test
