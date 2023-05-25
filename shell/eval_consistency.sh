#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}
DATA_PATH=${DATA_PATH:-data}
MODELS_PATH=${MODELS_PATH:-models}

for model in ComplEx SquaredComplEx TypedSquaredComplEx
do
    if [ $model = ComplEx ]
    then
      runs="OAdam_LR0.001_B5000_GNone_R10 OAdam_LR0.001_B5000_GNone_R50 OAdam_LR0.001_B5000_GNone_R200 OAdam_LR0.001_B5000_GNone_R1000"
    elif [ $model = SquaredComplEx ]
    then
      runs="OAdam_LR0.001_B1000_GNone_R10 OAdam_LR0.001_B2000_GNone_R50 OAdam_LR0.001_B5000_GNone_R200 OAdam_LR0.001_B5000_GNone_R1000"
    elif [ $model = TypedSquaredComplEx ]
    then
      runs="OAdam_LR0.001_B5000_GNone_R10 OAdam_LR0.001_B2000_GNone_R50 OAdam_LR0.001_B2000_GNone_R200 OAdam_LR0.001_B2000_GNone_R1000"
    else
      echo "Something is wrong" && exit 1
    fi
    python -m eval.consistency "${MODELS_PATH}" ogbl-biokg $model --data-path "${DATA_PATH}" \
      --exp-id "PLL" --run-names "${runs}" --device $DEVICE
    echo
done
