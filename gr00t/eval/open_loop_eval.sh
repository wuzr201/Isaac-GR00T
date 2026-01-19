# bash gr00t/eval/open_loop_eval.sh

export DATASET_PATH="/home/wuzr/lab/codebase/Isaac-GR00T/demo_data/lerobot_v20/4322/mm_pick_4322_green_64"
export CHECKPOINT_PATH="/home/wuzr/lab/codebase/Isaac-GR00T/outputs/kuavo_biped_finetune_joint_relative/checkpoint-15000"

uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $CHECKPOINT_PATH \
    --traj-ids 0 \
    --action-horizon 16  # ensure this is within the delta_indices of action's modality config.