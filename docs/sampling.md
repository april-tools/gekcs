# Reproducing Quality of Samples Results

First, set up the project and download and preprocess the data sets as showed in [README.md](../README.md).
The section **Training the Models** below lists the commands that trains GeKC models used in this experiment.

Then, you need to download the pre-trained embeddings of state-of-the-art KGE models from
the [repository](https://github.com/facebookresearch/ssl-relation-prediction#pretrained-embeddings)
of the [paper](https://openreview.net/forum?id=Qa3uS3H7-Le)
"Relation Prediction as an Auxiliary Training Objective for Improving Multi-Relational Graph Representations".
These models will be used to extract latent representations of triple (see our paper).
The downloaded files have to be saved in the directory containg the other models, as showed below for each data set.
```
path/to/saved/models/
|- FB15K-237/
   |- 1vsALL-Rel/
      |- OAdagrad_LR0.1_B500_GN3_R1000_L0.05_WR4.0/
        |- best_valid.pt
|- WN18RR/
   |- 1vsALL-Rel/
     |- OAdagrad_LR0.1_B100_GN3_R1000_L0.1_WR0.05/
       |- best_valid.pt
|- ogbl-biokg/
   |- 1vsALL-Rel/
     |- OAdagrad_LR0.1_B500_GN3_R1000_L0.01_WR0.25/
       |- best_valid.pt
```

Executing the script ```shell/eval_sampling.sh``` will print out the _Kernel Triple Distance_ (KTD)
(see the paper) scores with standard deviations as showed in the main paper and supplementary material.
Note that you need to specify the directory containing the models,
and optionally you can specify the device to use as showed in the following example.
```shell
MODELS_PATH=path/to/saved/models DEVICE=cuda:0 bash shell/eval_sampling.sh
```

## Training the Models

The following commands will train GeKC models required to replicate the results regarding sampling triples.

### FB15K-237

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
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_ll True
```

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model SquaredCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_ll True
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_ll True
```

### WN18RR

#### CP<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model NNegCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model NNegCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_ll True
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
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model SquaredCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredCP --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_ll True
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_ll True
```

### ogbl-biokg

#### CP<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model NNegCP --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model NNegCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.01 --score_ll True
```

#### ComplEx<sup>+</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.01 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model NNegComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.01 --score_ll True
```

#### CP<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredCP --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredCP --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_ll True
```

#### ComplEx<sup>2</sup> (PLL and MLE)

```shell
python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

```shell
python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 2000 --learning_rate 0.001 --score_ll True
```
