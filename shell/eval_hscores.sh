#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}
DATA_PATH=${DATA_PATH:-data}
MODELS_PATH=${MODELS_PATH:-models}

python -m eval.hscores "${MODELS_PATH}" --models "ComplEx" \
    --data-path $DATA_PATH --datasets "FB15K-237 WN18RR ogbl-biokg" --exp-id "PLL" \
    --runs "OAdam_LR0.001_B500_GNone_R1000
            OAdam_LR0.001_B500_GNone_R1000
            OAdam_LR0.001_B5000_GNone_R1000" \
    --y-range "1e-2 1.3" --x-range "-5.0 16.0" \
    --device $DEVICE
