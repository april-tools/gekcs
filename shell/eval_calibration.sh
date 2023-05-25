#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}
DATA_PATH=${DATA_PATH:-data}
MODELS_PATH=${MODELS_PATH:-models}

for baseline in CP ComplEx
do
  gekc_models="NNeg$baseline Squared$baseline"
  for expid in MLE PLL
  do
    if [ $baseline = CP ]
    then
      baseline_runs="OAdam_LR0.001_B5000_GNone_R1000;OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.001_B5000_GNone_R1000"
      if [ $expid = PLL ]
      then
        gekc_runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B1000_GNone_R1000;OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.01_B500_GNone_R1000 OAdam_LR0.001_B5000_GNone_R1000"
      else
        gekc_runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B1000_GNone_R1000;OAdam_LR0.01_B2000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.01_B500_GNone_R1000 OAdam_LR0.001_B5000_GNone_R1000"
      fi
    else
      baseline_runs="OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.001_B5000_GNone_R1000"
      if [ $expid = PLL ]
      then
        gekc_runs="OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B2000_GNone_R1000;OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.01_B500_GNone_R1000 OAdam_LR0.001_B5000_GNone_R1000"
      else
        gekc_runs="OAdam_LR0.01_B2000_GNone_R1000 OAdam_LR0.001_B500_GNone_R1000;OAdam_LR0.01_B5000_GNone_R1000 OAdam_LR0.001_B1000_GNone_R1000;OAdam_LR0.01_B500_GNone_R1000 OAdam_LR0.001_B5000_GNone_R1000"
      fi
    fi
    for chart_type in step line
    do
      python -m eval.calibration "${MODELS_PATH}" --datasets "FB15K-237 WN18RR ogbl-biokg" --data-path $DATA_PATH \
        --baseline-model $baseline --baseline-exp-id "PLL" --baseline-runs $baseline_runs \
        --gekc-models "$gekc_models" --gekc-exp-ids "$expid" --gekc-runs "${gekc_runs}" \
        --calibration-funcs "sigmoid minmax" --chart-type $chart_type --device $DEVICE \
        --enable-legend --enable-y-label
    done
  done
done
