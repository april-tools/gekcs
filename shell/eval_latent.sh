#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}
DATA_PATH=${DATA_PATH:-data}
MODELS_PATH=${MODELS_PATH:-models}

for dataset in FB15K-237 WN18RR ogbl-biokg
do
  if [ $dataset = FB15K-237 ]
  then
    ref_run="OAdagrad_LR0.1_B500_GN3_R1000_L0.05_WR4.0"
    flags="--manifold-alpha 0.07"
    runs="OAdam_LR0.001_B1000_GNone_R1000"
  elif [ $dataset = WN18RR ]
  then
    ref_run="OAdagrad_LR0.1_B100_GN3_R1000_L0.1_WR0.05"
    flags="--manifold-alpha 0.35"
    runs="OAdam_LR0.001_B500_GNone_R1000"
  elif [ $dataset = ogbl-biokg ]
  then
    ref_run="OAdagrad_LR0.1_B500_GN3_R1000_L0.01_WR0.25"
    flags="--manifold-alpha 0.09"
    runs="OAdam_LR0.001_B2000_GNone_R1000"
  else
    echo "Something is wrong" && exit 1
  fi
  python -m eval.sampling "${MODELS_PATH}" $dataset --data-path "${DATA_PATH}" \
    --ref-exp-id "1vsALL-Rel" --ref-run $ref_run \
    --eval-gekc-models "SquaredComplEx" --gekc-exp-ids "MLE" \
    --gekc-runs "${runs}" --device $DEVICE \
    --n-samples 25000 --plot-embeddings $flags
  echo -e '\n'
done
