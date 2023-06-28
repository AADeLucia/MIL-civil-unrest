#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="Minerva-CUT"

OUTPUT_DIR="${MINERVA_HOME}/models/instance_model"
OUTPUT_DIR="/data/aadelucia/minerva_instance_models"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

BATCH=128
python "${MINERVA_HOME}/code/train_instance_model.py" \
--overwrite_output_dir \
--n_trials 50 \
--num_train_epochs 20 \
--patience 3 \
--save_total_limit 3 \
--output_dir "${OUTPUT_DIR}" \
--logging_dir "${LOG_DIR}" \
--per_device_train_batch_size ${BATCH} \
--per_device_eval_batch_size ${BATCH} \
--warmup_steps 100 \
--logging_strategy "epoch" \
--log_on_each_node 0 \
--log_level "info" \
--save_strategy "epoch" \
--evaluation_strategy "epoch" \
--optim "adamw_torch" \
--dataloader_num_workers 4 \
--dataloader_drop_last True \
--seed 42 \
--load_best_model_at_end True \
--metric_for_best_model "loss" \
--dataloader_pin_memory False \
--ddp_find_unused_parameters False \
--report_to wandb

if [ $? -ne 0 ]
then
  echo "Error"
  exit 1
fi

echo "Done done"
