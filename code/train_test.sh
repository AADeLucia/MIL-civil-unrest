#!/bin/bash
# Similar to train.sh but uses a smaller dataset for debugging purposes
# Also only uses 1 GPU

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT="Minerva"
export WANDB_RUN_GROUP="Test-Min10"

# torchrun isn't working
# python -m torch.distributed.launch
# --sharded_ddp zero_dp_3 works for training but not inference
# -m torch.distributed.launch --nproc_per_node 2
# save+eval 1000, 10 epochs, 8 accumulation steps
DATA_DIR="${MINERVA_HOME}/data/premade_mil/minimum_10"
LOG_STEP=100
RATIOS=( 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.0 )
# Original was 1e-5
LR=10
for k in "${RATIOS[@]}"
do
  OUTPUT_DIR="${MINERVA_HOME}/models/test/ratio_${k}"
  LOG_DIR="${OUTPUT_DIR}/logs"
  mkdir -p "${LOG_DIR}"

  echo "Training model with key instance ratio ${k}. Saving to ${OUTPUT_DIR}"

  python "${MINERVA_HOME}/code/train_mil.py" \
    --instance_model "${MINERVA_HOME}/models/minerva_instance_models" \
    --dataset_dir "${DATA_DIR}" \
    --sample_instances False \
    --finetune_instance_model False \
    --run_name "test-${k}" \
    --key_instance_ratio "${k}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOG_DIR}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate "${LR}" \
    --num_train_epochs 2 \
    --num_tweets_per_day 100 \
    --logging_strategy "steps" \
    --log_on_each_node 0 \
    --logging_steps ${LOG_STEP} \
    --log_level "info" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --optim "adamw_torch" \
    --load_best_model_at_end True \
    --metric_for_best_model "f1" \
    --dataloader_num_workers 4 \
    --dataloader_drop_last True \
    --seed 42 \
    --dataloader_pin_memory False \
    --ddp_find_unused_parameters False \
    --report_to wandb \
    --do_train True \
    --do_predict True \

  if [ $? -ne 0 ]
  then
    echo "Error"
    exit 1
  fi

done

echo "Done done"
