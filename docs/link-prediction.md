# Reproducing Link Prediction and Additional Results

First, set up the project and download and preprocess the data sets as showed in [README.md](../README.md).

## Link Prediction

The following commands will reproduce the results relative to link prediction showed in the paper.
By default, all the results are written into Tensorboard files and stored in ```tboard-runs/```.
However, this can be changed by specifying a different path with the parameter ```--tboard_dir```,
and a different device can be selected by specifying it with the parameter ```--device```
(the default one is ```cuda```).
In addition, the best models (based on metrics computed on validation data) are saved in ```models/``` by default.
However, this can be changed by specifying a different path with the parameter ```--model_cache_path```.

The next sections show the commands that reproduce the experimental results regarding link prediction,
i.e., mean-reciprocal-rank (MRR), hits@k scores and, for the proposed GeKCs only, also the average log-likelihood. 
All the models have been trained using the pseudo-log-likelihood objective (PLL),
while the proposed GeKCs have been trained also by maximum-likelihood estimation (MLE) (see the paper for details).

Note that all the experiments were executed on a RTX A6000 with 48 GiB of memory.
Nevertheless, for FB15K-237 and WN18RR a GPU with 16Gib of memory should be sufficient.

### FB15K-237

#### CP (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model CP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### ComplEx (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model ComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### CP<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model NNegCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model NNegCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_ll True
```

#### ComplEx<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.01 --score_ll True
```

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model SquaredCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_ll True
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_ll True
```

### WN18RR

#### CP (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model CP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### ComplEx (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model ComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### CP<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model NNegCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model NNegCP --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.01 --score_ll True
```

#### ComplEx<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_ll True
```

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model SquaredCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_ll True
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_ll True
```

### ogbl-biokg

#### CP (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model CP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### ComplEx (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model ComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### CP<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model NNegCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model NNegCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.01 --score_ll True
```

#### ComplEx<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.01 --score_ll True
```

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_ll True
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_ll True
```

### ogbl-wikikg2

#### ComplEx (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-wikikg2 --model ComplEx --rank 100 --optimizer Adam --batch_size 250 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

#### ComplEx<sup>2</sup> (PLL)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-wikikg2 --model SquaredComplEx --rank 100 --optimizer Adam --batch_size 10000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

## Parameters Distillation

After pre-training CP (resp. ComplEx) it is possible to distil its parameters into CP<sup>2</sup> (resp. ComplEx<sup>2</sup>)
and possibily fine-tune it to retrieve a tractable generative models over triples (see the parper for details).
To do so, first run one of the commands above to train CP (or ComplEx) with the best hyperparameters found.
Then, distil a GeCK model by running one of the following commands,
which specifies the run id of the corresponding model to distil (either CP or ComplEx).

By default, the model to be distilled are assumed to be in ```models/```,
which however can be  modified by specifying a different path with the parameter ```--model_cache_path```.

### FB15K-237

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model SquaredCP --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True \
  --distil_model True --distil_run "OAdam_LR0.001_B5000_GNone_R1000"
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_ll True \
  --distil_model True --distil_run "OAdam_LR0.001_B5000_GNone_R1000"
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.0001 --score_lhs True --score_rel True --score_rhs True \
  --distil_model True --distil_run "OAdam_LR0.001_B500_GNone_R1000"
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_ll True \
  --distil_model True --distil_run "OAdam_LR0.001_B500_GNone_R1000"
```

### WN18RR

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model SquaredCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.0001 --score_lhs True --score_rel True --score_rhs True \
  --distil_model True --distil_run "OAdam_LR0.001_B500_GNone_R1000"
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_ll True \
  --distil_model True --distil_run "OAdam_LR0.001_B500_GNone_R1000"
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_lhs True --score_rel True --score_rhs True \
  --distil_model True --distil_run "OAdam_LR0.001_B500_GNone_R1000"
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_ll True \
  --distil_model True --distil_run "OAdam_LR0.001_B500_GNone_R1000"
```

### ogbl-biokg

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True \
  --distil_model True --distil_run "OAdam_LR0.001_B5000_GNone_R1000"
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_ll True \
  --distil_model True --distil_run "OAdam_LR0.001_B5000_GNone_R1000"
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True \
  --distil_model True --distil_run "OAdam_LR0.001_B5000_GNone_R1000"
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.0001 --score_ll True \
  --distil_model True --distil_run "OAdam_LR0.001_B5000_GNone_R1000"
```

## Reproduce Calibration Plots

Here we show how to reproduce the calibration plots showed in the supplementary material.
After training the models for link prediction (see above sections),
run the script ```shell/eval_calibration.sh``` to reproduce the plots, as showed in the following example.

```shell
MODELS_PATH=path/to/saved/models DEVICE=cuda:0 bash shell/eval_calibration.sh
```
