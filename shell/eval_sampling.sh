#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}
DATA_PATH=${DATA_PATH:-data}
MODELS_PATH=${MODELS_PATH:-models}

for dataset in FB15K-237 WN18RR ogbl-biokg
do
  for expid in MLE PLL
  do
    if [ $dataset = FB15K-237 ]
    then
      ref_run="OAdagrad_LR0.1_B500_GN3_R1000_L0.05_WR4.0"
      if [ $expid = PLL ]
      then
        runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000 OAdam_LR0.001_B1000_GNone_R1000"
      else
        runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B1000_GNone_R1000 OAdam_LR0.001_B1000_GNone_R1000"
      fi
    elif [ $dataset = WN18RR ]
    then
      ref_run="OAdagrad_LR0.1_B100_GN3_R1000_L0.1_WR0.05"
      if [ $expid = PLL ]
      then
        runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B5000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000"
      else
        runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B5000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000"
      fi
    elif [ $dataset = ogbl-biokg ]
    then
      ref_run="OAdagrad_LR0.1_B500_GN3_R1000_L0.01_WR0.25"
      if [ $expid = PLL ]
      then
        runs="OAdam_LR0.01_B1000_GNone_R1000 OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000"
      else
        runs="OAdam_LR0.01_B500_GNone_R1000 OAdam_LR0.01_B500_GNone_R1000 OAdam_LR0.001_B2000_GNone_R1000 OAdam_LR0.001_B2000_GNone_R1000"
      fi
    else
      echo "Something is wrong" && exit 1
    fi
    python -m eval.sampling "${MODELS_PATH}" $dataset --data-path "${DATA_PATH}" \
      --ref-exp-id "1vsALL-Rel" --ref-run $ref_run \
      --eval-gekc-models "NNegCP NNegComplEx SquaredCP SquaredComplEx" --gekc-exp-ids $expid \
      --gekc-runs "${runs}" --eval-uniform --eval-nnmfaug --device $DEVICE
    echo -e '\n'
  done
done
