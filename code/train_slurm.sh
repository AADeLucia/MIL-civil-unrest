#!/bin/bash
#SBATCH -A mdredze80_gpu
#SBATCH --partition ica100
#SBATCH --array=0-10%4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --qos=qos_gpu
#SBATCH --job-name="mil-topk"
#SBATCH --output="logs/%x.%A_%a.out" # Path to store logs

# Set environment
module load anaconda
module load cuda/11.6.0
conda activate minerva-proj
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Choose parameter for this task
RATIOS=( 1.0 0.3 0.5 0.7 0.0 0.1 0.2 0.4 0.6 0.8 0.9 )
k=${RATIOS[${SLURM_ARRAY_TASK_ID}]}
LOG_STEP=100
SAVE_STEP=1000
DATA_DIR="${MINERVA_HOME}/data/premade_mil/minimum_10"
MODEL_DIR="/home/adeluci2/data-mdredze1/adeluci2/models"
BASE_OUTPUT_DIR="${MODEL_DIR}/finetune_shuffle"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/ratio_${k}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
export WANDB_PROJECT="Minerva"
export WANDB_RUN_GROUP="MIL-Min10-v2"

# Parameter settings
EPOCHS=50
LR=1e-5
TRAIN_BATCH=5
#ACC_STEPS=$((EFFECTIVE_BATCH/TRAIN_BATCH))
ACC_STEPS=1
EVAL_BATCH=10
BAG_SIZE=100
PATIENCE=20
WARMUP=100
# Start training
echo "Training model with key instance ratio ${k}. Saving to ${OUTPUT_DIR}"
# srun python -m torch.distributed.launch --nproc_per_node 4 "${MINERVA_HOME}/code/train_mil.py" \
torchrun --nproc_per_node 4 "${MINERVA_HOME}/code/train_mil.py" \
  --dataset_dir "${DATA_DIR}" \
  --instance_model "${MODEL_DIR}/minerva_instance_models/hp_best" \
  --sample_instances True \
  --patience "${PATIENCE}" \
  --save_total_limit 3 \
  --warmup_steps "${WARMUP}" \
  --finetune_instance_model True \
  --run_name "finetune-${k}" \
  --key_instance_ratio "${k}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --per_device_train_batch_size "${TRAIN_BATCH}" \
  --gradient_accumulation_steps "${ACC_STEPS}" \
  --per_device_eval_batch_size "${EVAL_BATCH}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --num_tweets_per_day "${BAG_SIZE}" \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_strategy "steps" \
  --log_on_each_node 0 \
  --logging_steps ${LOG_STEP} \
  --log_level "info" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEP}" \
  --evaluation_strategy "steps" \
  --eval_steps "${SAVE_STEP}" \
  --optim "adamw_torch" \
  --load_best_model_at_end True \
  --metric_for_best_model "f1" \
  --dataloader_num_workers 4 \
  --dataloader_drop_last True \
  --seed 42 \
  --dataloader_pin_memory False \
  --ddp_find_unused_parameters False \
  --resume_from_checkpoint False

if [ $? -ne 0 ]
then
  echo "Error"
  exit 1
fi

