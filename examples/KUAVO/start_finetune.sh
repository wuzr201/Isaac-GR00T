set -x -e

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_DIR="/home/wuzr/lab/codebase/Isaac-GR00T/demo_data/lerobot_v20/mix"

DATASET_NAMES=(
  "mm_pick_4322_green_64"
  "mm_pick_4322_green_86"
  "mm_pick_4322_green_106"
  "mm_unpack_left_4322_green_64"
  "mm_unpack_left_4322_green_86"
  "mm_unpack_left_4322_green_106"
  "mm_unpack_right_4322_green_64"
  "mm_unpack_right_4322_green_86"
  "mm_unpack_right_4322_green_106"
  "mm_pick_4611_green_64"
  "mm_pick_4611_green_86"
  "mm_pick_4611_green_106"
  "mm_unpack_left_4611_green_64"
  "mm_unpack_left_4611_green_86"
  "mm_unpack_left_4611_green_106"
  "mm_unpack_right_4611_green_64"
  "mm_unpack_right_4611_green_86"
  "mm_unpack_right_4611_green_106"
  "mm_pick_4633_green_64"
  "mm_pick_4633_green_86"
  "mm_pick_4633_green_106"
  "mm_unpack_left_4633_green_64"
  "mm_unpack_left_4633_green_86"
  "mm_unpack_left_4633_green_106"
  "mm_unpack_right_4633_green_64"
  "mm_unpack_right_4633_green_86"
  "mm_unpack_right_4633_green_106"
)

DATASET_PATHS=""
for name in "${DATASET_NAMES[@]}"; do
  if [[ -z "$DATASET_PATHS" ]]; then
    DATASET_PATHS="${DATASET_DIR}/${name}"
  else
    DATASET_PATHS="${DATASET_PATHS}:${DATASET_DIR}/${name}"
  fi
done

export DATASET_PATHS

uv run torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29700 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path  $DATASET_PATHS \
    --modality_config_path /home/wuzr/lab/codebase/Isaac-GR00T/demo_data/lerobot_v20/modality_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir ./outputs/kuavo_biped_joint_relative_multi_head \
    --save_steps 5000 \
    --save_total_limit 4 \
    --max_steps 30000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 64 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --shard_size 100 \
    --use_wandb