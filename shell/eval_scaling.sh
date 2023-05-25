#!/bin/bash

export PYTHONPATH=src
DEVICE=${DEVICE:-cuda}

python -m eval.scaling ogbl-wikikg2 --models "ComplEx NNegComplEx SquaredComplEx" \
  --batch-sizes "100 200 500 1000 2000 5000" --entities-fractions "0.125 0.25 0.5 1.0" \
  --models-hparams "rank=100;rank=100;rank=100" --num-iterations 25 --burnin-iterations 10 \
  --device $DEVICE
