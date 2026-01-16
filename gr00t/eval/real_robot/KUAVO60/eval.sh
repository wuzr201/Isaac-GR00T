export CHECKPOINT_PATH=""   # replace with your checkpoint path


# uv run python gr00t/eval/open_loop_eval.py \
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python gr00t/eval/real_robot/KUAVO60/eval.py \
    --model-path $CHECKPOINT_PATH \
    --action-horizon 12  # ensure this is within the delta_indices of action's modality config.