#!/bin/bash
#
# EBM Bimanual Composer Evaluation Script
# 
# Usage:
#   bash scripts/rlbench/eval_ebm_bimanual_composer.sh
#   bash scripts/rlbench/eval_ebm_bimanual_composer.sh <experiment_name>
#   bash scripts/rlbench/eval_ebm_bimanual_composer.sh /path/to/checkpoint.pth
#   nohup bash scripts/rlbench/eval_ebm_bimanual_composer.sh exp_name > eval.log 2>&1 &
#

get_output_experiment_name() {
    if [ -n "$EXPERIMENT_NAME" ]; then
        echo "$EXPERIMENT_NAME"
        return 0
    fi

    local parent_pid=$PPID
    local parent_cmdline=$(cat /proc/$parent_pid/cmdline 2>/dev/null | tr '\0' ' ')

    if [[ "$parent_cmdline" =~ \>\ *([^\ ]+\.log) ]]; then
        local log_file="${BASH_REMATCH[1]}"
        log_file=$(basename "$log_file")
        log_file="${log_file%.log}"
        echo "$log_file"
        return 0
    fi

    local parent_processes=$(ps -o pid,ppid,cmd -e | grep "eval_ebm_bimanual_composer.sh" | grep -v grep)
    if [ -n "$parent_processes" ]; then
        local log_file=$(echo "$parent_processes" | grep -oP '>\s*\K[^\s]+\.log' | head -1)
        if [ -n "$log_file" ]; then
            log_file=$(basename "$log_file")
            log_file="${log_file%.log}"
            echo "$log_file"
            return 0
        fi
    fi

    echo "ebm_bimanual_composer_eval_$(date +%Y%m%d_%H%M%S)"
}

get_experiment_and_checkpoint() {
    local experiment_name=""
    local checkpoint_path=""

    if [ "$#" -gt 0 ]; then
        if [[ "$1" == *.pth ]]; then
            checkpoint_path="$1"
            experiment_name=$(basename $(dirname $(dirname "$1")))
        else
            experiment_name="$1"
            if [ -f "./ebm_results/$experiment_name/energy_based_flow_matching/best.pth" ]; then
                checkpoint_path="./ebm_results/$experiment_name/energy_based_flow_matching/best.pth"
            else
                checkpoint_path="./ebm_results/$experiment_name/energy_based_flow_matching/last.pth"
            fi
        fi
    else
        local latest_dir=$(ls -t ./ebm_results/ 2>/dev/null | head -1)
        if [ -n "$latest_dir" ]; then
            experiment_name="$latest_dir"
            if [ -f "./ebm_results/$latest_dir/energy_based_flow_matching/best.pth" ]; then
                checkpoint_path="./ebm_results/$latest_dir/energy_based_flow_matching/best.pth"
            else
                checkpoint_path="./ebm_results/$latest_dir/energy_based_flow_matching/last.pth"
            fi
        else
            echo "ERROR: No experiments found in ebm_results/"
            exit 1
        fi
    fi

    echo "$experiment_name|$checkpoint_path"
}

EXPERIMENT_AND_CHECKPOINT=$(get_experiment_and_checkpoint "$@")
SOURCE_EXPERIMENT_NAME=$(echo "$EXPERIMENT_AND_CHECKPOINT" | cut -d'|' -f1)
CHECKPOINT=$(echo "$EXPERIMENT_AND_CHECKPOINT" | cut -d'|' -f2)
OUTPUT_EXPERIMENT_NAME=$(get_output_experiment_name)

echo "Source experiment: $SOURCE_EXPERIMENT_NAME"
echo "Output experiment: $OUTPUT_EXPERIMENT_NAME"

# Bimanual tasks
exp=ebm_bimanual_composer
tasks=(
    bimanual_push_box
    bimanual_lift_ball
    bimanual_dual_push_buttons
    bimanual_pick_plate
    bimanual_put_item_in_drawer
    bimanual_put_bottle_in_fridge
    bimanual_handover_item
    bimanual_pick_laptop
    bimanual_straighten_rope
    bimanual_sweep_to_dustpan
    bimanual_lift_tray
    bimanual_handover_item_easy
    bimanual_take_tray_out_of_oven
)

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    ls -la ./ebm_results/ 2>/dev/null || echo "  No experiments found"
    exit 1
fi

checkpoint_alias="ebm_bimanual_composer_model"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Evaluation settings
max_tries=2
max_steps=25
headless=true
collision_checking=false
seed=0

# Dataset
data_dir="/path/to/peract2_raw/peract2_test"
dataset=Peract2Bimanual
image_size=256,256

# Model architecture (must match training config)
model_type=ebm_bimanual_composer
bimanual=true
prediction_len=1
backbone=clip
embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3
num_shared_attn_layers=4
relative_action=false
rotation_format="quat_xyzw"
denoise_timesteps=5
denoise_model=rectified_flow

export CUDA_VISIBLE_DEVICES="0"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# EBM parameters
use_ebm_composition=true
freeze_flow_weights=false
fps_subsampling_factor=5
enable_energy_in_eval=true

# Adaptive denoising
enable_adaptive_denoising=false
min_denoise_steps=1
max_denoise_steps=5
energy_threshold_low=4.0
energy_threshold_high=15.0
enable_early_stopping=false
early_stop_check_interval=1

# Coordination constraints
enable_coordination_constraints=true
coord_constraint_weight=0.1
coord_weight_hidden_dim=64
min_ee_distance=0.001
min_joint_distance=0.001

eval_output_dir="eval_logs/$exp/$OUTPUT_EXPERIMENT_NAME/seed$seed"
mkdir -p "$eval_output_dir"

echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Tasks: ${#tasks[@]}"
echo "Output: $eval_output_dir"
echo ""

if [ -n "$OUTPUT_EXPERIMENT_NAME" ]; then
    export PROCESS_NAME="$OUTPUT_EXPERIMENT_NAME"
fi

eval "$(conda shell.bash hook)"
conda activate energyaction

export PYTHONPATH="${PYTHONPATH}:."
export COPPELIASIM_ROOT="/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

cd "$(dirname "$0")/../.."

echo "Starting evaluation of ${#tasks[@]} bimanual tasks..."
echo ""

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    echo "Evaluating task $((i+1))/${num_ckpts}: ${tasks[$i]}"

    xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint "$CHECKPOINT" \
        --task ${tasks[$i]} \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --headless $headless \
        --collision_checking $collision_checking \
        --seed $seed \
        --data_dir "$data_dir" \
        --dataset $dataset \
        --image_size $image_size \
        --output_file "$eval_output_dir/${tasks[$i]}/eval.json" \
        --model_type $model_type \
        --bimanual $bimanual \
        --prediction_len $prediction_len \
        --backbone $backbone \
        --fps_subsampling_factor $fps_subsampling_factor \
        --embedding_dim $embedding_dim \
        --num_attn_heads $num_attn_heads \
        --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
        --num_history $num_history \
        --num_shared_attn_layers $num_shared_attn_layers \
        --relative_action $relative_action \
        --rotation_format $rotation_format \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model \
        --enable_energy_in_eval $enable_energy_in_eval \
        --enable_adaptive_denoising $enable_adaptive_denoising \
        --min_denoise_steps $min_denoise_steps \
        --max_denoise_steps $max_denoise_steps \
        --energy_threshold_low $energy_threshold_low \
        --energy_threshold_high $energy_threshold_high \
        --enable_early_stopping $enable_early_stopping \
        --early_stop_check_interval $early_stop_check_interval \
        --enable_coordination_constraints $enable_coordination_constraints \
        --coord_constraint_weight $coord_constraint_weight \
        --coord_weight_hidden_dim $coord_weight_hidden_dim \
        --min_ee_distance $min_ee_distance \
        --min_joint_distance $min_joint_distance

    if [ $? -eq 0 ]; then
        echo "✓ ${tasks[$i]} completed"
    else
        echo "✗ ${tasks[$i]} failed"
    fi
    echo ""
done

echo "Collecting results..."
python online_evaluation_rlbench/collect_results.py --folder "$eval_output_dir/"

if [ $? -eq 0 ]; then
    echo "✓ Evaluation completed. Results: $eval_output_dir"
else
    echo "✗ Result collection failed"
    exit 1
fi

exit 0
