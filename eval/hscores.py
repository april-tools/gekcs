import argparse
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from eval.utils import setup_data_loader, load_model, setup_tueplots, format_model_name
from kbc.datasets import Dataset
from kbc.models import KBCModel
from kbc.gekc_models import SquaredKBCModel
from kbc.utils import set_seed


def plot_histogram_scores(
        model: KBCModel,
        data_loader: DataLoader,
        ax: plt.Axes,
        n_bins: int = 100,
        desc: Optional[str] = None,
        color: Optional[str] = None
):
    y_scores = torch.empty(len(data_loader.dataset.tensors[0]), dtype=torch.float32)
    for i, (batch,) in tqdm(enumerate(data_loader)):
        batch = batch.to(device)
        y_score = model.score(batch)
        if isinstance(model, SquaredKBCModel):
            y_score = torch.log(y_score + model.eps)
        y_score = y_score.squeeze(dim=1)
        y_scores[i * batch_size:i * batch_size + batch_size] = y_score.cpu()
    y_scores = y_scores.numpy()
    assert np.all(np.isfinite(y_scores))
    y_min, y_max = np.quantile(y_scores, q=[0.0025, 0.9975])
    ax.hist(y_scores, bins=n_bins, range=(y_min, y_max), density=True, alpha=0.4, label=desc, color=color)


parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Plot Histogram of Scores (in log-space)"
)
parser.add_argument(
    'models_path', type=str, help="The root path containing the models"
)
parser.add_argument(
    '--models', required=True, type=str, help="The model names"
)
parser.add_argument(
    '--datasets', required=True, type=str, help="The evaluation datasets, separated by space"
)
parser.add_argument(
    '--exp-id', type=str, default='PLL', help="The experiment ID of the models"
)
parser.add_argument(
    '--runs', type=str, default="", help="The run names for each dataset and for each model "
                                         "(separated by space and semicolon, respectively)"
)
parser.add_argument(
    '--model-filename', type=str, default='best_valid.pt', help="The models' filenames"
)
parser.add_argument(
    '--data-path', type=str, default='data', help="The path containing the data"
)
parser.add_argument(
    '--n-bins', type=int, default=50, help="The number of bins"
)
parser.add_argument(
    '--seed', type=int, default=42, help="The random seed"
)
parser.add_argument(
    '--device', type=str, default='cuda', help="The device id"
)
parser.add_argument(
    '--batch-size', type=int, default=1000, help="The batch size"
)
parser.add_argument(
    '--x-range', type=str, default="", help="The fixed range of values on the x-axis, empty to disable"
)
parser.add_argument(
    '--y-range', type=str, default="", help="The fixed range of values on the y-axis, empty to disable"
)
parser.add_argument(
    '--show-model-name', action='store_true', help="Whether to show the names of the models"
)

if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Set some useful constants
    batch_size = args.batch_size
    models = args.models.split()
    datasets = args.datasets.split()
    runs = list(map(str.split, args.runs.split(';')))
    nrows, ncols = len(models), 2
    setup_tueplots(nrows, ncols, hw_ratio=0.55)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
    if args.x_range:
        x_min, x_max = tuple(map(float, args.x_range.split(" ")))
    if args.y_range:
        y_min, y_max = tuple(map(float, args.y_range.split(" ")))

    for i, m in enumerate(models):
        for j, (r, d) in enumerate(zip(runs[i], datasets)):
            # Load dataset
            ds = Dataset(d, device, reciprocal=False, data_path=args.data_path)
            n_entities, n_relations, _ = ds.get_shape()
            train_data = ds.get_split('train')
            valid_data = ds.get_split('valid')
            test_data = ds.get_split('test')
            target_data = valid_data
            negative_data, _ = ds.load_negatives(split='valid')
            negative_data_loader = setup_data_loader(negative_data, batch_size=batch_size)
            target_data_loader = setup_data_loader(valid_data, batch_size=batch_size)

            # Load the model and plot histograms
            model = load_model(
                args.models_path, d, n_entities, n_relations,
                m, args.exp_id, r, args.model_filename,
                device=device, filter_inverted_relations=args.exp_id not in ['PLL', 'MLE']
            )
            plot_histogram_scores(model, target_data_loader, ax[i, 0], n_bins=args.n_bins, desc=d, color=f'C{j}')
            plot_histogram_scores(model, negative_data_loader, ax[i, 1], n_bins=args.n_bins, desc=d, color=f'C{j}')
            if args.x_range:
                ax[i, 0].set_xlim(x_min, x_max)
            if args.y_range:
                ax[i, 1].set_ylim(y_min, y_max)
        ax[i, 0].set_yscale('log')
        ax[i, 1].set_yscale('log')
        if args.show_model_name:
            ax[i, 0].set_ylabel(format_model_name(m))
    ax[-1, 0].set_xlabel(r"$\phi(s,r,o)$")
    ax[-1, 1].set_xlabel(r"$\phi(s,r,\widehat{o})$")

    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.09 if args.show_model_name else 0.06, 0.995))

    filename = "hscores-{}-{}".format(
        '-'.join(list(map(str.lower, datasets))),
        '-'.join(list(map(str.lower, models)))
    )
    plt.savefig(f'{filename}.pdf')
