# Standalone script, use scripts/rlbench/hiveformer_datagen.sh
# to generate and package as well
DATA_PATH=hiveformer_raw/

seed=0
variation=0
variation_count=1
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

num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python data_generation/generate.py \
          --save_path ${DATA_PATH} \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 100 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed ${seed}
done
