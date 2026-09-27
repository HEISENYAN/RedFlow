#!/bin/bash

TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"

    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"

    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_burger_fries"

    "place_can_basket"
    "place_cans_plasticbox"
    "place_container_plate"
    "place_dual_shoes"
    "place_empty_cup"

    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle"
    "shake_bottle_horizontally"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
    "stamp_seal"
    "turn_switch"
)
PARTIAL_TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
)
AWR_TASKS=(
    "adjust_bottle"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "beat_block_hammer"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
)
policy_name=cot


echo "Policy: ${policy_name}"

#source .venv/bin/activate
cd ../..  # Move to root

#rollout
PYTHONWARNINGS=ignore::UserWarning \
python script/multi_gpu_eval_rollout_scheduler.py \
    --eval_mode eval \
    --config policy/${policy_name}/deploy_policy.yml \
    --task_names "${TASKS[@]}" \
    --gpu_num 16 \
    --batch_size 50 \
    --seed_base 0 \
    --overrides \
    --task_config demo_clean \
    --train_config_name pi0_robotwin \
    --model_name robotwin_50_clean \
    --ckpt_setting robotwin_50_clean \
    --policy_name ${policy_name} \
    --checkpoint_base_dir ./policy/cot/checkpoints \
    --assets_base_dir ./ \
    --checkpoint_id 40000

# eval
# PYTHONWARNINGS=ignore::UserWarning \
# uv run script/multi_gpu_eval_rollout_scheduler.py \
#     --eval_mode eval \
#     --config policy/${policy_name}/deploy_policy.yml \
#     --task_names "${TASKS[@]}" \
#     --gpu_num 2 \
#     --batch_size 2 \
#     --seed_base 0 \
#     --overrides \
#     --task_config demo_clean \
#     --train_config_name pi0_base_aloha_robotwin_full_torch \
#     --model_name robotwin_aloha_lerobot \
#     --ckpt_setting robotwin_aloha_lerobot \
#     --policy_name ${policy_name}
