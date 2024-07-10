# How to Turn Your Knowledge Graph Embeddings into Generative Models

This repository contains the code for reproducing the experiments of the paper
[_"How to Turn Your Knowledge Graph Embeddings into Generative Models"_](https://openreview.net/forum?id=RSGNGiB1q4), which has been accepted at NeurIPS 2023 as oral (top 0.6%).

Inspired by state-of-the-art models of link prediction (e.g., ComplEx),
we introduce a novel class of tractable generative models of triples in a knowledge graph (called GeKCs)
whose implementation can be found in this repository.
This repository extends an [existing codebase](https://github.com/facebookresearch/ssl-relation-prediction)
by introducing GeKCs and scripts used to reproduce our experiments.

## Project Structure

The repository is structured as follows.
The file ```requirements.txt``` contains all the required Python dependencies, which can be installed by ```pip```.
The directory ```src``` contains the module ```kbc``` containing all the code and in ```tests```
we store sanity checks that can be run by executing ```pytest``` at the root level.
The directories ```eval``` and ```shell``` contains evaluation scripts.
In particular, the shell scripts in ```shell``` execute Python scripts in ```eval``` with the correct parameters to
reproduce the results and figures of the paper, once the models have been trained or downloaded (see below sections).
Finally, ```econfigs``` contains the config files for performing a hyperparameters grid search.

## Downloading and Preprocessing Datasets

You can download the datasets from [here](https://github.com/villmow/datasets_knowledge_embedding)
(only tran/valid/test.txt files are needed) and put them in a new ```src_data``` directory
having the following structure.
```
src_data/
|- FB15K-237/
   |- train.txt
   |- valid.txt
   |- test.txt
|- ...
```

After that execute the following command by specifying the datasets you need as follows.
```shell
python -m kbc.preprocess src_data --datasets "FB15K-237 WN18RR ogbl-biokg ogbl-wikikg2"
```
This will create a new directory ```data``` (by default) which will store the data sets.

### Additional Preprocessing

If you wish to reproduce the experiments on sampling triples and predictions calibration (see the paper)
run the following command instead.
```shell
python -m kbc.preprocess src_data --datasets "FB15K-237 WN18RR ogbl-biokg" \
  --save-negatives --nnmf
```
This will create two additional files for each dataset that will contain negative triples
and the necessary features to construct the [NNMFAug baseline](https://arxiv.org/abs/2110.13205). 

## Training Models from Scratch

The script ```src/kbc/experiment.py``` executes an experiment,
and the results can be saved as [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) files
or uploaded on [Weights & Biases](https://wandb.ai/site),
while model weights can be saved by specifying the right flags.

For instance, the following command trains ComplEx on ```ogbl-biokg```
using the pseudo-log-likelihood (PLL) objective (see the paper for details).
```shell
python -m kbc.experiment --experiment_id PLL --tboard_dir "tboard-runs" --model_cache_path "models" \
  --dataset ogbl-biokg --model ComplEx --rank 1000 --batch_size 1000 --optimizer Adam --learning_rate 1e-3 \
  --score_lhs True --score_rel True --score_rhs True --device cuda
```
The results can be then visualized with Tensorbord by pointing it to the specified directory, i.e., ```tboard-runs/```,
and the models are saved into the ```models/``` directory.
To upload the results (not the models) on Weights & Biases then you have to specify the following flags.
```
  --wandb True --wandb_project org/myproject
```

Another example is the following command, which trains a ComplEx<sup>2</sup> model with the same hyperparameters above
but using the maximum-likelihood estimation (MLE) objective (see the paper for details).
```shell
python -m kbc.experiment --experiment_id MLE --tboard_dir "tboard-runs" --model_cache_path "models" \
  --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --batch_size 1000 --optimizer Adam --learning_rate 1e-3 \
  --score_ll True --device cuda
```
All the implemented models can be found in ```kbc.models``` and ```kbc.gekc_models``` modules.

### Model Checkpoints

For traditional KGE models (e.g., CP or ComplEx), a single checkpoint will be saved to disk during training
if ```--model_cache_path``` is specified.
The best model found according to the mean-reciprocal-rank computed on validation data
can be found under ```<model_cache_path>/<dataset>/<exp_id>/<run_id>```
for some experiment id and run id (i.e., an alias for the chosen hyperparameters).

For GeCKs two model checkpoints will be saved to disk during training
if ```--model_cache_path``` is specified (like in the previous command):
1. the best model found according to the MRR computed on validation data
   (under ```<model_cache_path>/<dataset>/<exp_id>/<run_id>/kbc/```);
2. the best model found according to the average log-likelihood computed on validation data
   (under ```<model_cache_path>/<dataset>/<exp_id>/<run_id>/gen/```).

### Hyperparameters Grid Search

To run a _grid_ of experiments, use the ```src/kbc/grid.py``` script
by specifying one of the config files in ```econfigs```, which contains all the needed settings.

For instance, to reproduce the grid search performed for the link prediction experiments (see the paper)
run the following command.
```shell
python -m kbc.grid econfigs/large-datasets.json
```
You can also specifiy multiple devices to be used in parallel as follows (e.g., multiple CUDA device IDs).
```shell
python -m kbc.grid econfigs/large-datasets.json --multi-devices "cuda:0 cuda:1 cuda:2"
```

## Reproducing Results

### Link Prediction and Additional Results

In [link-prediction.md](docs/link-prediction.md) we list the commands and hyperparameters to
reproduce the results of the link prediction experiments.
In addition, we show how to replicate the results relative to distilling and fine-tuning the proposed GeKCs
from pre-trained KGE models (i.e., CP and ComplEx), and how to plot calibration curves of the models.

### Integration od Domain Constraints

In [domain-constraints.md](docs/domain-constraints.md) we list the commands, hyperparameters and instructions to
reproduce the results about (i) how many triples that violate domain constraints are predicted by the models
and (ii) how helpful for link prediction the integration of domain constraints in GeKCs is. 

### Quality of Sampled Triples Results

In [sampling.md](docs/sampling.md) we list the commands, hyperparameters and instructions to
reproduce the results regarding the quality of sampled triples.
