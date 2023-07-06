#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
BASE_OUTPUT_DIR="${MINERVA_HOME}/data/premade_mil"

CPU=20
MAX=-1
THRESHOLDS=( 10 100 0 )
THRESHOLDS=( 0 )
for threshold in "${THRESHOLDS[@]}"
do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/minimum_${threshold}"
  # CHANGE THIS
  OUTPUT_DIR="${BASE_OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  python "${MINERVA_HOME}/code/create_dataset.py" \
    --output-dir  "${OUTPUT_DIR}" \
    --train-files "${MINERVA_HOME}/data/tweets_en/201[456]_.*.gz" \
    --val-files "${MINERVA_HOME}/data/tweets_en/2017_.*.gz" \
    --test-files "${MINERVA_HOME}/data/tweets_en/201[89]_.*.gz" \
    --acled-file "${MINERVA_HOME}/data/2014-01-01-2020-01-01_acled_reduced_all.csv" \
    --max-instances ${MAX} \
    --min-instances ${threshold} \
    --n-cpu ${CPU} \
    --add-instance-scores \
    --instance-model "${MINERVA_HOME}/models/minerva_instance_models" \
    --only-add-instance-scores \
    --batchsize 2000

  if [ $? -ne 0 ]
  then
    echo "Failed on threshold ${threshold}"
    exit 1
  fi

done
