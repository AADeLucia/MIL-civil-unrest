#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
BASE_OUTPUT_DIR="${MINERVA_HOME}/data/premade_mil"

MAX=1000
for threshold in 10 100
do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/minimum_${threshold}"
  mkdir -p "${OUTPUT_DIR}"

  python "${MINERVA_HOME}/code/create_dataset.py" \
    --output-dir  "${OUTPUT_DIR}" \
    --train-files "${MINERVA_HOME}/data/tweets_en/201[456]_.*.gz" \
    --val-files "${MINERVA_HOME}/data/tweets_en/2017_.*.gz" \
    --test-files "${MINERVA_HOME}/data/tweets_en/201[89]_.*.gz" \
    --acled-file "${MINERVA_HOME}/data/2014-01-01-2020-01-01_acled_reduced_all.csv" \
    --max-instances ${MAX} \
    --min-instances ${threshold} \
    --n-cpu 20 \
    --add-instance-scores \
    --instance-model "${MINERVA_HOME}/models/minerva_instance_models" \

  if [ $? -ne 0 ]
  then
    echo "Failed on threshold ${threshold}"
    exit 1
  fi

done
