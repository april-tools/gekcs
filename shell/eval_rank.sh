#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}
DATA_PATH=${DATA_PATH:-data}
MODELS_PATH=${MODELS_PATH:-models}

complex_runs="OAdam_LR0.001_B5000_GNone_R10 OAdam_LR0.001_B5000_GNone_R50 OAdam_LR0.001_B5000_GNone_R200 OAdam_LR0.001_B5000_GNone_R1000"
squared_complex_runs="OAdam_LR0.001_B1000_GNone_R10 OAdam_LR0.001_B2000_GNone_R50 OAdam_LR0.001_B5000_GNone_R200 OAdam_LR0.001_B5000_GNone_R1000"
typed_squared_complex_runs="OAdam_LR0.001_B5000_GNone_R10 OAdam_LR0.001_B2000_GNone_R50 OAdam_LR0.001_B2000_GNone_R200 OAdam_LR0.001_B2000_GNone_R1000"
runs="${complex_runs};${squared_complex_runs};${typed_squared_complex_runs}"

python -m eval.rank "${MODELS_PATH}" "ogbl-biokg" --data-path "${DATA_PATH}" \
  --models "ComplEx SquaredComplEx TypedSquaredComplEx" \
  --run-names "${runs}" --exp-ids "PLL" --device $DEVICE
