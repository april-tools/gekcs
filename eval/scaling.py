import argparse
import gc
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import rcParams
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from eval.utils import setup_tueplots, format_model_name, from_bytes_to_gib
from kbc.datasets import Dataset
from kbc.distributions import Categorical, TwinCategorical
from kbc.engines import setup_model
from kbc.losses import setup_loss
from kbc.models import MODELS_NAMES, KBCModel, ComplEx, CP
from kbc.gekc_models import TSRL_MODELS_NAMES, TractableKBCModel, SquaredCP, SquaredComplEx, NNegCP, NNegComplEx
from kbc.utils import set_seed

parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Evaluate Pseudo-Log-Likelihood Objective Efficiency"
)
parser.add_argument(
    'dataset', type=str, help="The evaluation dataset"
)
parser.add_argument(
    '--data-path', type=str, default='data', help="The path containing the data"
)
parser.add_argument(
    '--num-iterations', type=int, default=1, help="The number of iterations per benchmark"
)
parser.add_argument(
    '--burnin-iterations', type=int, default=1, help="Burnin iterations (additional to --num-iterations)"
)
parser.add_argument(
    '--models', type=str, default="", help="List of models to evaluate, separated by space"
)
parser.add_argument(
    '--models-hparams', type=str,
    default="rank=2000;rank=200 rank_r=100",
    help="List of hyperparameters (A=b) for each model specified in --models separaterd by semicolons",
)
parser.add_argument(
    '--device', type=str, default='cuda', help="The device id"
)
parser.add_argument(
    '--batch-sizes', type=str, default="100 500", help="A list of batch sizes separated by space"
)
parser.add_argument(
    '--entities-fractions', type=str, default="0.1 1.0", help="The fractions of entities to take into account"
)
parser.add_argument(
    '--output-file', type=str, default='pll-scaling.pdf', help="The PDF output filepath"
)
parser.add_argument(
    '--min-bubble-radius', type=float, default=20.0, help="Bubble sizes minimum"
)
parser.add_argument(
    '--scale-bubble-radius', type=float, default=1.0, help="Bubble sizes scaler"
)
parser.add_argument(
    '--exp-bubble-radius', type=float, default=1.75, help="The exponent for computing the bubble sizes"
)
parser.add_argument(
    '--seed', type=int, default=42, help="The seed for reproducibility"
)


def bubble_size(s: float, a: float = 0.0, m: float = 1.0, p: float = 2.0, inverse: bool = False) -> float:
    if inverse:
        return ((s - a) ** (1.0 / p)) / m
    return a + ((m * s) ** p)


def eval_time_by_batch_size(
        model: KBCModel,
        dataset: np.ndarray,
        batch_size: int,
        num_iterations: int,
        burnin_iterations: int
):
    # Setup the data loader
    total_num_iterations = burnin_iterations + num_iterations
    assert batch_size * total_num_iterations < len(dataset), "Number of iterations is too large for this dataset and batch size"
    ordering = np.random.permutation(len(dataset))
    dataset = dataset[ordering]
    dataset = TensorDataset(torch.from_numpy(dataset[:batch_size * total_num_iterations]))
    data_loader = DataLoader(dataset, batch_size, drop_last=True)
    try:
        mu_time, mu_memory = run_benchmark(data_loader, model, burnin_iterations=burnin_iterations)
    except torch.cuda.OutOfMemoryError:
        mu_time, mu_memory = np.nan, np.nan
    return mu_time, mu_memory


def eval_memory_by_num_entities(
        model: KBCModel,
        dataset: np.ndarray,
        n_entities: int,
        filtered_n_entities: int,
        num_iterations: int,
        burnin_iterations: int,
        batch_size: int = 500
) -> Tuple[float, float]:
    filtered_entities = np.random.choice(n_entities, size=filtered_n_entities, replace=False)
    new_entities_map = np.arange(n_entities)
    new_entities_map[filtered_entities] = np.arange(filtered_n_entities)
    filtered_dataset = dataset[np.isin(dataset[:, 0], filtered_entities) & np.isin(dataset[:, 2], filtered_entities)].copy()
    filtered_dataset[:, 0] = new_entities_map[filtered_dataset[:, 0]]
    filtered_dataset[:, 2] = new_entities_map[filtered_dataset[:, 2]]
    dataset = filtered_dataset
    assert len(dataset) > 0
    total_num_iterations = burnin_iterations + num_iterations
    while len(dataset) < total_num_iterations * batch_size:
        dataset = np.concatenate([dataset, filtered_dataset])
    ordering = np.random.permutation(len(dataset))
    dataset = dataset[ordering][:total_num_iterations * batch_size]
    assert len(dataset) == (total_num_iterations * batch_size)

    if isinstance(model, CP):
        model.entity = nn.Embedding(filtered_n_entities, model.entity.embedding_dim)
    elif isinstance(model, ComplEx):
        model.embeddings[0] = nn.Embedding(filtered_n_entities, model.embeddings[0].embedding_dim)
    elif isinstance(model, (SquaredCP, SquaredComplEx)):
        model.ent_embeddings = nn.Embedding(filtered_n_entities, model.ent_embeddings.embedding_dim)
    elif isinstance(model, NNegCP):
        model.ent_embeddings = Categorical(filtered_n_entities, model.ent_embeddings.batch_size)
    elif isinstance(model, NNegComplEx):
        model.ent_embeddings = TwinCategorical(filtered_n_entities, model.ent_embeddings.batch_size)
    else:
        raise NotImplementedError()
    model.to(device)

    # Setup the data loader
    assert len(dataset) > 0
    dataset = TensorDataset(torch.from_numpy(dataset))
    data_loader = DataLoader(dataset, batch_size, drop_last=True)

    try:
        mu_time, mu_memory = run_benchmark(data_loader, model, burnin_iterations=burnin_iterations)
    except torch.cuda.OutOfMemoryError:
        mu_time, mu_memory = np.nan
    return mu_time, mu_memory


def run_benchmark(data_loader: DataLoader, model: KBCModel, burnin_iterations: int = 1) -> Tuple[float, float]:
    # Setup losses and a dummy optimizer (only used to free gradient tensors)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    kbc_loss_func = setup_loss('LCWA+ce')  # loss for non-tractable KBC models
    tsrl_loss_func = setup_loss('NLL')  # loss for tractable KBC models

    elapsed_times = list()
    gpu_memory_peaks = list()
    for batch, in data_loader:
        # Run GC manually and then disable it
        gc.collect()
        gc.disable()
        # Reset peak memory usage statistics
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()        # Synchronize CUDA operations
        batch = batch.to(device)
        torch.cuda.synchronize(device)  # Make sure the batch is already loaded (do not take into account this!)
        start_time = time.perf_counter()
        if isinstance(model, TractableKBCModel):
            _, (rhs_probs, rel_probs, lhs_probs) = model.forward(
                batch, score_rhs=True, score_rel=True, score_lhs=True, score_ll=False
            )
            loss = tsrl_loss_func(rhs_probs) + tsrl_loss_func(rel_probs) + tsrl_loss_func(lhs_probs)
        else:
            (rhs_scores, rel_scores, lhs_scores), _ = model.forward(
                batch, score_rhs=True, score_rel=True, score_lhs=True
            )
            loss = kbc_loss_func(rhs_scores, batch[:, 2]) + \
                   kbc_loss_func(rel_scores, batch[:, 1]) + \
                   kbc_loss_func(lhs_scores, batch[:, 0])
        loss.backward(retain_graph=False)  # Free the autodiff graph
        torch.cuda.synchronize(device)     # Synchronize CUDA Kernels before measuring time
        end_time = time.perf_counter()
        gpu_memory_peaks.append(from_bytes_to_gib(torch.cuda.max_memory_allocated(device)))
        gc.enable()            # Enable GC again
        optimizer.zero_grad()  # Free gradients tensors
        gc.collect()           # Manual GC
        elapsed_times.append(end_time - start_time)

    # Discard burnin iterations and compute averages
    elapsed_times = elapsed_times[burnin_iterations:]
    gpu_memory_peaks = gpu_memory_peaks[burnin_iterations:]
    mu_time = np.mean(elapsed_times).item()
    print(f"Mean time: {mu_time} -- Std. time: {np.std(elapsed_times)}")
    mu_memory = np.mean(gpu_memory_peaks).item()
    return mu_time, mu_memory


if __name__ == '__main__':
    args = parser.parse_args()
    batch_sizes = sorted(map(int, args.batch_sizes.split()))
    entities_fractions = sorted(map(float, args.entities_fractions.split()))

    # Set device and the seed
    device = torch.device(args.device)
    set_seed(args.seed)

    # Load data and setup tensor dataset
    ds = Dataset(args.dataset, device, reciprocal=False, data_path=args.data_path)
    n_entities, n_relations, _ = ds.get_shape()
    dataset = ds.get_split('train')
    prog_n_entities = list(map(lambda x: int(x * n_entities), entities_fractions))
    # Set models' identifiers and their hyperparameters
    models = args.models.split()
    if not all(map(lambda m: m in MODELS_NAMES + TSRL_MODELS_NAMES, models)):
        raise ValueError("There is an unknown model name")
    hparams = args.models_hparams.split(";")
    assert len(hparams) == len(models), "You should specify a list of hparams for each model"

    nrows, ncols = 1, 2
    setup_tueplots(nrows, ncols, hw_ratio=0.45)
    fig, ax = plt.subplots(nrows, ncols, sharey=True)
    def _bubble_size(s, inverse=False):
        return bubble_size(
            s, a=args.min_bubble_radius,
            m=args.scale_bubble_radius, p=args.exp_bubble_radius,
            inverse=inverse
        )

    scatter_plots = dict()
    markers = ['o', 'o', 'o', 's', '^', 'D', 'h']
    for idx, (m, hp) in enumerate(zip(models, hparams)):
        hp = list(map(lambda hv: tuple(hv.split("=")), hp.split()))
        assert all(map(lambda hv: len(hv) == 2, hp)), "Invalid hparams format"
        hp = list(map(lambda hv: (hv[0], float(hv[1]) if '.' in hv[1] else int(hv[1])), hp))
        conf = dict(hp)
        conf.update({'model': m, 'init_scale': 1e-3})

        print(f"Benchmarking {m} ...")
        model = setup_model((n_entities, n_relations, n_entities), conf).to(device)
        bench_bs_results, bench_es_results = list(), list()
        for bs in batch_sizes:
            mu_time, mu_memory = eval_time_by_batch_size(
                model, dataset, batch_size=bs,
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations
            )
            bench_bs_results.append((mu_time, mu_memory))
        for ef in entities_fractions:
            filtered_n_entities = int(ef * n_entities)
            mu_time, mu_memory = eval_memory_by_num_entities(
                model, dataset, n_entities, filtered_n_entities,
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations
            )
            bench_es_results.append((mu_time, mu_memory))
        desc = format_model_name(m)
        bench_bs_results = list(filter(lambda t: np.isfinite(t[0]), bench_bs_results))
        bench_es_results = list(filter(lambda t: np.isfinite(t[0]), bench_es_results))
        sc_bs = ax[0].scatter(
            batch_sizes[:len(bench_bs_results)], list(map(lambda t: t[0], bench_bs_results)),
            color=f'C{idx}', alpha=.5, s=list(map(lambda t: _bubble_size(t[1]), bench_bs_results)),
            marker='o', label=desc
        )
        ax[0].scatter(
            batch_sizes[:len(bench_bs_results)], list(map(lambda t: t[0], bench_bs_results)),
            color='k', alpha=.6, s=1, marker=markers[idx]
        )
        sc_es = ax[1].scatter(
            prog_n_entities, list(map(lambda t: t[0], bench_es_results)),
            color=f'C{idx}', alpha=.5, s=list(map(lambda t: _bubble_size(t[1]), bench_es_results)),
            marker='o', label=desc
        )
        ax[1].scatter(
            prog_n_entities, list(map(lambda t: t[0], bench_es_results)),
            color='k', alpha=.6, s=1, marker=markers[idx]
        )
        scatter_plots[m] = {'bs': sc_bs, 'es': sc_es}
        del model

    ax[0].set_ylabel('Time per batch ($s$)')
    ax[0].annotate('Batch size', xy=(1, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']), ha='right', va='top',
                   xycoords='axes fraction', textcoords='offset points')
    ax[1].annotate('Num. of entities', xy=(1, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']), ha='right', va='top',
                   xycoords='axes fraction', textcoords='offset points')
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xticks([2 * (10 ** 5), 10 ** 6])
    ax[0].margins(x=0.175, y=0.275)
    ax[1].margins(x=0.175, y=0.275)
    ax[0].set_ylim(bottom=-0.05)
    ax[1].set_ylim(bottom=-0.05)
    ax[0].grid(linestyle='--', alpha=0.3, linewidth=.5)
    ax[1].grid(linestyle='--', alpha=0.3, linewidth=.5)
    models_legend = ax[0].legend(
        loc='upper right', bbox_to_anchor=(1.0, 1.0),
        labelspacing=0.4
    )
    for i in range(len(models_legend.legend_handles)):
        models_legend.legend_handles[i].set_sizes([20])
    ax[0].add_artist(models_legend)
    ax[0].text(-0.62, 0.45, "GPU Memory (GiB)", rotation=90, va='center', transform=ax[0].transAxes)
    ax[0].legend(
        loc='upper right', bbox_to_anchor=(-0.275, 1.05),
        labelspacing=2.0, frameon=False,
        *scatter_plots[models[0]]['bs'].legend_elements(
            prop='sizes', func=lambda s: _bubble_size(s, inverse=True),
            alpha=0.6, fmt="{x:.0f}", num=4
        ),
        handletextpad=1.0
    )
    fig.savefig(args.output_file)
    plt.show()
