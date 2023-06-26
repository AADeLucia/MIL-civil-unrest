#!/bin/bash

OUTPUT_DIR="${MINERVA_HOME}/data/premade_mil"
mkdir -p "${OUTPUT_DIR}"

python "${MINERVA_HOME}/code/create_dataset.py" \
  --output-dir  "${OUTPUT_DIR}" \
  --train-files "${MINERVA_HOME}/data/tweets_en/201[456]_.*.gz" \
  --val-files "${MINERVA_HOME}/data/tweets_en/2017_.*.gz" \
  --test-files "${MINERVA_HOME}/data/tweets_en/201[89]_.*.gz" \
  --acled-file "${MINERVA_HOME}/data/2014-01-01-2020-01-01_acled_reduced_all.csv" \
  --max-instances 1000 \
  --n-cpu 20 \

