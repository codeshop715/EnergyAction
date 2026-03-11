#!/bin/bash
#
# Stage 1: Single-Arm 3D Flow Actor Training Script
#
# Train a single-arm denoise3d model with Rectified Flow on RLBench single-arm data.
# The trained checkpoint will be used to initialize both left and right arms
# in the EBM Bimanual Composer (Stage 2).
#
# Usage:
#   bash scripts/rlbench/train_single_arm.sh
#
# With nohup (recommended for long training):
#   nohup bash scripts/rlbench/train_single_arm.sh > train_single_arm.log 2>&1 &
#

# ======================== Environment ========================
eval "$(conda shell.bash hook)"
conda activate energyaction

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:."

cd "$(dirname "$0")/../.."

# ======================== Data Paths ========================
# Single-arm RLBench data (Peract format, Zarr)
TRAIN_DATA_DIR="/path/to/zarr_datasets/peract/train.zarr/"
EVAL_DATA_DIR="/path/to/zarr_datasets/peract/val.zarr/"
TRAIN_INSTRUCTIONS="instructions/peract/instructions.json"
VAL_INSTRUCTIONS="instructions/peract/instructions.json"

# ======================== Training Config ========================
DATASET="Peract"
MODEL_TYPE="denoise3d"

# Hyperparameters
BATCH_SIZE=64
LR=1e-4
LR_SCHEDULER="constant"
WD=1e-10
TRAIN_ITERS=100001
VAL_FREQ=20

# Model architecture
BACKBONE="clip"
EMBEDDING_DIM=120
NUM_ATTN_HEADS=8
NUM_VIS_INSTR_ATTN_LAYERS=3
NUM_HISTORY=3
NUM_SHARED_ATTN_LAYERS=4
FPS_SUBSAMPLING_FACTOR=4

# Denoising
DENOISE_TIMESTEPS=5
DENOISE_MODEL="rectified_flow"

# Output
MAIN_DIR="Peract"
RUN_LOG_DIR="${MODEL_TYPE}-${DATASET}-C${EMBEDDING_DIM}-B${BATCH_SIZE}-lr${LR}-${LR_SCHEDULER}-H${NUM_HISTORY}-${DENOISE_MODEL}"

echo "============================================"
echo "  Stage 1: Single-Arm 3D Flow Actor Training"
echo "============================================"
echo "Dataset:    $DATASET"
echo "Model:      $MODEL_TYPE"
echo "LR:         $LR"
echo "Iterations: $TRAIN_ITERS"
echo "Output:     train_logs/$MAIN_DIR/$RUN_LOG_DIR"
echo ""

# ======================== Launch Training ========================
torchrun --nproc_per_node 1 --master_port $RANDOM \
    main.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --eval_data_dir "$EVAL_DATA_DIR" \
    --train_instructions "$TRAIN_INSTRUCTIONS" \
    --val_instructions "$VAL_INSTRUCTIONS" \
    --dataset "$DATASET" \
    --num_workers 4 \
    --batch_size $BATCH_SIZE \
    --batch_size_val $BATCH_SIZE \
    --chunk_size 1 \
    --memory_limit 8 \
    --exp_log_dir "$MAIN_DIR" \
    --run_log_dir "$RUN_LOG_DIR" \
    --val_freq $VAL_FREQ \
    --eval_only false \
    --lr $LR \
    --backbone_lr 1e-6 \
    --lr_scheduler "$LR_SCHEDULER" \
    --wd $WD \
    --train_iters $TRAIN_ITERS \
    --use_compile false \
    --use_ema false \
    --lv2_batch_size 1 \
    --model_type "$MODEL_TYPE" \
    --bimanual false \
    --keypose_only true \
    --pre_tokenize true \
    --backbone "$BACKBONE" \
    --finetune_backbone false \
    --finetune_text_encoder false \
    --fps_subsampling_factor $FPS_SUBSAMPLING_FACTOR \
    --embedding_dim $EMBEDDING_DIM \
    --num_attn_heads $NUM_ATTN_HEADS \
    --num_vis_instr_attn_layers $NUM_VIS_INSTR_ATTN_LAYERS \
    --num_history $NUM_HISTORY \
    --num_shared_attn_layers $NUM_SHARED_ATTN_LAYERS \
    --workspace_normalizer_buffer 0.05 \
    --relative_action false \
    --rotation_format "quat_xyzw" \
    --denoise_timesteps $DENOISE_TIMESTEPS \
    --denoise_model "$DENOISE_MODEL" \
    --max_demos_per_task -1

echo ""
echo "Training complete. Checkpoint saved to: train_logs/$MAIN_DIR/$RUN_LOG_DIR/"
echo "Use this checkpoint as LEFT_PRETRAINED_PATH and RIGHT_PRETRAINED_PATH in Stage 2."
