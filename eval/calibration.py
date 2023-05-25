import argparse
import os
from typing import Optional

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve

from eval.utils import setup_data_loader, load_model, setup_tueplots, format_model_name
from kbc.datasets import Dataset
from kbc.models import KBCModel
from kbc.gekc_models import SquaredKBCModel, TractableKBCModel
from kbc.utils import set_seed


def compute_calibration_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        desc: str,
        ax: plt.Axes,
        chart_type: str = 'step',
        n_bins: int = 10,
        color: Optional[str] = None,
        marker: Optional[str] = None,
        strategy: str = 'quantile'
):
    if chart_type == 'step':
        bins = np.arange(n_bins + 1) / float(n_bins)
        pos_hist, pos_breaks = np.histogram(y_pred[y_true == 1], bins)
        neg_hist, _ = np.histogram(y_pred[y_true == 0], bins)
        pos_frac = np.nan_to_num(pos_hist / (pos_hist + neg_hist))
        avg_probs = np.array([
            np.mean(y_pred[(a <= y_pred) & (y_pred < b)])
            for (a, b) in zip(pos_breaks, pos_breaks[1:])
        ])
        ece = np.nanmean(np.abs(avg_probs - pos_frac))
        desc = "{} - ECE: {:.3f}".format(desc, ece)
        ax.step(
            pos_breaks,
            np.concatenate([pos_frac, [pos_frac[-1]]]),
            label=desc, color=color, alpha=0.8, linewidth=2.0, where='post'
        )
    elif chart_type == 'line':
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy=strategy
        )
        ece = np.mean(np.abs(prob_true - prob_pred))
        ax.plot([0, 1], [0, 1], "k:")
        marker = 's' if marker is None else marker
        ax.plot(prob_pred, prob_true, marker + '-', alpha=0.8, label='{} - ECE: {:.3f}'.format(desc, ece))
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")


def evaluate_model_calibration(
        model: KBCModel,
        data_loader: DataLoader,
        desc: Optional[str],
        ax: plt.Axes,
        calibration_func: str = 'sigmoid',
        chart_type: str = 'step',
        color: Optional[str] = None,
        marker: Optional[str] = None,
        strategy: str = 'quantile',
        train_data_loader: Optional[DataLoader] = None
) -> float:
    batch_size = data_loader.batch_size
    n_samples = len(data_loader.dataset)
    y_pred = torch.empty(n_samples, dtype=torch.float32)
    y_true = torch.empty(n_samples, dtype=torch.int64)

    if calibration_func in ['maxexp', 'minmax']:
        min_y_score = torch.tensor(torch.inf, dtype=torch.float32, device=device)
        max_y_score = torch.tensor(-torch.inf, dtype=torch.float32, device=device)
        for (x_batch,) in train_data_loader:
            x_batch = x_batch.to(device)
            y_score = model.score(x_batch)
            if isinstance(model, SquaredKBCModel):
                y_score = torch.log(y_score + model.eps)
            max_y_score = torch.max(max_y_score, torch.max(y_score))
            min_y_score = torch.min(min_y_score, torch.min(y_score))

    for i, (x_batch, y_batch) in tqdm(enumerate(data_loader), desc=f"Evaluating {desc}"):
        x_batch = x_batch.to(device)
        if isinstance(model, TractableKBCModel):
            if calibration_func == 'sigmoid':
                y_score = model.score(x_batch)
                if isinstance(model, SquaredKBCModel):
                    y_score = torch.log(y_score + model.eps)
                y_score = torch.sigmoid(y_score)
            elif calibration_func == 'maxcon':
                qs = model.get_queries(x_batch, target='rhs')
                cs = model.get_candidates(0, n_entities, target='rhs', device=device)
                y_score = model.eval_circuit_all(qs, cs)
                if isinstance(model, SquaredKBCModel):
                    y_score = torch.log(y_score + model.eps)
                max_ll = torch.max(y_score, dim=1)[0]
                idx = torch.arange(len(y_score), device=y_score.device)
                y_score = y_score[idx, x_batch[:, 2]]
                y_score = torch.exp(y_score - max_ll).unsqueeze(dim=1)
            elif calibration_func == 'maxexp':
                y_score = model.score(x_batch)
                if isinstance(model, SquaredKBCModel):
                    y_score = torch.log(y_score + model.eps)
                y_score = torch.exp(y_score - max_y_score)
            elif calibration_func == 'minmax':
                y_score = model.score(x_batch)
                if isinstance(model, SquaredKBCModel):
                    y_score = torch.log(y_score + model.eps)
                y_score = (y_score - min_y_score) / (max_y_score - min_y_score)
                y_score = torch.clamp(y_score, min=0.0, max=1.0)
            else:
                raise ValueError(f"Unknown calibration function {calibration_func}")
        else:
            if calibration_func == 'sigmoid':
                y_score = model.score(x_batch)
                y_score = torch.sigmoid(y_score)
            elif calibration_func == 'maxcon':
                (y_score, _, _), _ = model.forward(x_batch)
                y_score = y_score - torch.logsumexp(y_score, dim=1, keepdim=True)
                max_ll = torch.max(y_score, dim=1)[0]
                idx = torch.arange(len(y_score), device=y_score.device)
                y_score = y_score[idx, x_batch[:, 2]]
                y_score = torch.exp(y_score - max_ll).unsqueeze(dim=1)
            elif calibration_func == 'maxexp':
                y_score = model.score(x_batch)
                y_score = torch.exp(y_score - max_y_score)
            elif calibration_func == 'minmax':
                y_score = model.score(x_batch)
                y_score = (y_score - min_y_score) / (max_y_score - min_y_score)
                y_score = torch.clamp(y_score, min=0.0, max=1.0)
            else:
                raise ValueError(f"Unknown calibration function {calibration_func}")
        y_pred[i * batch_size:i * batch_size + batch_size] = y_score.cpu().squeeze(dim=1)
        y_true[i * batch_size:i * batch_size + batch_size] = y_batch

    y_pred = y_pred.numpy().astype(np.float32, copy=True)
    y_true = y_true.numpy().astype(np.int64, copy=True)
    assert np.all(np.isfinite(y_pred))
    assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
    assert np.all((y_true == 0) | (y_true == 1))

    return compute_calibration_plot(
        y_true, y_pred, desc,
        ax=ax, chart_type=chart_type, color=color, marker=marker, strategy=strategy
    )


parser = argparse.ArgumentParser(
    description="Evaluate KGE models predictions calibration Script"
)
parser.add_argument(
    'models_path', type=str, help="The root path containing the models"
)
parser.add_argument(
    '--datasets', required=True, type=str, help="The evaluation datasets, separated by space"
)
parser.add_argument(
    '--baseline-model', type=str, default='ComplEx', choices=['CP', 'ComplEx'], help="The baseline model"
)
parser.add_argument(
    '--baseline-exp-id', type=str, default='PLL', help="The experiment ID of the baseline model"
)
parser.add_argument(
    '--baseline-runs', required=True, type=str, help="List of baseline model run names, separated by semicolon"
)
parser.add_argument(
    '--model-filename', type=str, default='best_valid.pt', help="The models' filenames"
)
parser.add_argument(
    '--data-path', type=str, default='data', help="The path containing the data"
)
parser.add_argument(
    '--seed', type=int, default=42, help="The random seed"
)
parser.add_argument(
    '--gekc-models', required=True, type=str, help="List of GeKC models to evaluate, separated by space"
)
parser.add_argument(
    '--gekc-exp-ids', type=str, default="PLL", help="List of experiments IDs of GeKC models, separated by space"
)
parser.add_argument(
    '--gekc-runs', required=True, type=str,
    help="List of GeKC run names, separated by space and separated by semicolon for each dataset"
)
parser.add_argument(
    '--device', type=str, default='cuda', help="The device id"
)
parser.add_argument(
    '--batch-size', type=int, default=100, help="The batch size"
)
parser.add_argument(
    '--neg-method', default='constraint-n', choices=['random', 'constraint-n'], help="The method of sampling negative triples"
)
parser.add_argument(
    '--calibration-funcs', default="sigmoid minmax",
    help="The list of calibration functions to use, each in {'maxcon', 'sigmoid', 'maxexp', 'minmax'}"
)
parser.add_argument(
    '--chart-type', default='step', choices=['step', 'line'], help="The type of the chart to plot"
)
parser.add_argument(
    '--chart-line-strategy', default='uniform', choices=['uniform', 'quantile'],
    help="The strategy of selecting bin edges when --chart-type is 'line'"
)
parser.add_argument(
    '--full-title', action='store_true', help="Whether to show the full title"
)
parser.add_argument(
    '--enable-x-label', action='store_true', help="Whether to show x-axis label"
)
parser.add_argument(
    '--enable-y-label', action='store_true', help="Whether to show x-axis label"
)
parser.add_argument(
    '--enable-legend', action='store_true', help="Whether to show the legend"
)
parser.add_argument(
    '--legend-bbox', type=str, default="0.0 -0.1", help="The bounding box of the legend"
)

if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Set some useful constants
    batch_size = args.batch_size
    models_path = args.models_path
    datasets = args.datasets.split()
    calibration_funcs = args.calibration_funcs.split()
    gekc_models = args.gekc_models.split()
    gekc_exp_ids = args.gekc_exp_ids.split()
    gekc_runs = list(map(lambda x: x.split(), args.gekc_runs.split(';')))
    baseline_runs = args.baseline_runs.split(';')
    nrows, ncols = len(calibration_funcs), len(datasets)
    setup_tueplots(nrows, ncols, rel_width=1.0, hw_ratio=1.25)
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False)

    markers = ['o', 's', '^', 'D', 'h', '*']
    calibration_confs = list(map(
        lambda x: (x[0], (x[1], enumerate(datasets))),
        enumerate(calibration_funcs)
    ))
    calibration_confs = [
        ((i, j), (cf, d)) for (i, (cf, ls)) in calibration_confs for (j, d) in ls
    ]  # A list of (i, j) indices and re-calibration function (i), dataset name (j)
    for (i, j), (cf, d) in calibration_confs:
        # Load dataset
        ds = Dataset(d, device, reciprocal=False, data_path=args.data_path)
        neg_triples, test_indices = ds.load_negatives()
        n_entities, n_relations, _ = ds.get_shape()
        train_data = ds.get_split('train')
        test_data = ds.get_split('test')
        test_data = test_data[test_indices]
        if test_data.shape[1] > 3:
            test_data = test_data[:, :3]
        x_data = np.concatenate([neg_triples, test_data], axis=0)
        y_data = np.concatenate([np.zeros(len(neg_triples)), np.ones(len(test_data))], axis=0)
        data_loader = setup_data_loader(x_data, y_data, batch_size=args.batch_size)
        train_data_loader = setup_data_loader(train_data, batch_size=args.batch_size)
        print(f"# of true triples: {len(test_data)}")
        print(f"# of negative triples: {len(neg_triples)}")

        # Load baseline model
        baseline_model = load_model(
            models_path, d, n_entities, n_relations, args.baseline_model,
            args.baseline_exp_id, baseline_runs[j], args.model_filename, device=device
        )
        # Evaluate the baseline model
        evaluate_model_calibration(
            baseline_model, data_loader,
            desc=f"{format_model_name(args.baseline_model)}",
            ax=ax[i, j], calibration_func=cf,
            chart_type=args.chart_type, color='C0', marker=markers[0],
            strategy=args.chart_line_strategy, train_data_loader=train_data_loader
        )
        del baseline_model

        # Evaluate GEKCs
        assert len(gekc_exp_ids) == 1 or len(gekc_exp_ids) == len(gekc_models), \
            "You should specify one experiment ID for each GeKC model, or only one"
        assert len(gekc_runs[j]) == len(gekc_models), \
            "You should specify one run name for each GeKC model"
        if len(gekc_exp_ids) == 1:
            gekc_exp_ids = gekc_exp_ids * len(gekc_models)
        for k, (m, e, r) in enumerate(zip(gekc_models, gekc_exp_ids, gekc_runs[j])):
            model = load_model(
                models_path, d, n_entities, n_relations,
                m, e, r, args.model_filename, device=device
            )
            evaluate_model_calibration(
                model, data_loader,
                desc=format_model_name(m), ax=ax[i, j], calibration_func=cf,
                chart_type=args.chart_type, color=f'C{k + 1}', marker=markers[k + 1],
                strategy=args.chart_line_strategy, train_data_loader=train_data_loader
            )
            del model

        ax[i, j].set_box_aspect(1.0)
        ax[i, j].set_ylim(-0.05, 1.05)
        ax[i, j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[i, j].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if args.enable_legend:
            if args.legend_bbox:
                bbox_to_anchor = list(map(float, args.legend_bbox.split()))
            else:
                bbox_to_anchor = None
            ax[i, j].legend(loc='upper left', bbox_to_anchor=bbox_to_anchor)
        else:
            ax[i, j].get_legend().remove()
        if args.enable_x_label:
            if args.chart_type == 'step':
                ax[i, j].set_xlabel('Calibration Scores Bins')
            elif args.chart_type == 'line':
                ax[i, j].set_xlabel('Mean Calibration Scores')
        else:
            ax[i, j].xaxis.label.set_visible(False)
        if j > 0:
            ax[i, j].set(ylabel=None)
    if args.enable_y_label:
        for i in range(len(calibration_funcs)):
            ax[i, 0].set_ylabel('Ratio of Positives')
    else:
        for i in range(len(calibration_funcs)):
            ax[i, 0].yaxis.label.set_visible(False)
    if args.full_title:
        for j, d in enumerate(datasets):
            ax[0, j].set_title(r"\textsf{" + d + r"} " + " - " + args.neg_method)
    else:
        for j, d in enumerate(datasets):
            ax[0, j].set_title(r"\textsf{" + d + r"}")

    filename = f'calibration-{args.baseline_model.lower()}'
    if args.chart_type == 'step':
        directory = f"calibration-results-{'-'.join(calibration_funcs)}-{args.chart_type}"
    elif args.chart_type == 'line':
        directory = f"calibration-results-{'-'.join(calibration_funcs)}-{args.chart_type}-{args.chart_line_strategy}"
    else:
        directory = 'calibration-results'
    gekc_exp_ids_s = list(map(str.lower, set(gekc_exp_ids)))
    directory = f"{directory}-{'-'.join(gekc_exp_ids_s)}"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, f'{filename}.pdf'))
