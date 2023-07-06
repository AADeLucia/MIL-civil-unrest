#!/bin/bash

DATA_DIR="${MINERVA_HOME}/data/premade_mil/minimum_10"
OUTPUT_DIR="${MINERVA_HOME}/models/baselines"
mkdir -p "${OUTPUT_DIR}"

python "${MINERVA_HOME}/code/run_baselines.py" \
  --dataset-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --models "rf" "lr" "random" "country-random"
