# Reproducing Integration of Domain Constraints Results

First, set up the project and download and preprocess the ```ogbl-biokg``` data set
as showed in [README.md](../README.md).
The section **Training the Models** below lists the commands that trains ComplEx,
ComplEx<sup>2</sup> and ComplEx<sup>2</sup> with domain constraints that are used in this experiments

Then, running the script ```shell/eval_consistency.sh``` will print out the _semantic consistency scores at k_ (Sem@k)
(see our paper).
Note that you need to specify the directory containing the models and the device, as showed in the following example.
```shell
MODELS_PATH=path/to/saved/models DEVICE=cuda:0 bash shell/eval_consistency.sh
```

Furthermore, to plot the line graph showing how the test Mean Reciprocal Rank (MRR) changes by increasing
the embedding size, run the script ```shell/eval_rank```, as showed in the following example.
```shell
MODELS_PATH=path/to/saved/models DEVICE=cuda:0 bash shell/eval_rank.sh
```

## Training the Models

### ComplEx

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model ComplEx --rank 10 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model ComplEx --rank 50 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model ComplEx --rank 200 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model ComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

### ComplEx<sup>2</sup>

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 10 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 50 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 200 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

### ComplEx<sup>2</sup> with Domain Constraints

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model TypedSquaredComplEx --rank 10 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model TypedSquaredComplEx --rank 50 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model TypedSquaredComplEx --rank 200 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model TypedSquaredComplEx --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```
