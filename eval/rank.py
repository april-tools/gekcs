import argparse

import matplotlib.pyplot as plt
import torch

from eval.utils import load_model, setup_tueplots, format_model_name, retrieve_best_hparams, build_run_id
from kbc.datasets import Dataset
from kbc.utils import average_metrics_entity


parser = argparse.ArgumentParser(
    description="Evaluate Link Prediction by rank ablation Script"
)
parser.add_argument(
    'models_path', type=str, help="The root path containing the models"
)
parser.add_argument(
    'dataset', type=str, help="The evaluation dataset"
)
parser.add_argument(
    '--models', required=True, type=str, help="The list of model names, separated by space"
)
parser.add_argument(
    '--exp-ids', type=str, default='1vsAll', help="The experiment ID of the models, separated by space"
)
parser.add_argument(
    '--run-names', required=True, type=str,
    help="The list of run names, separated by semicolon for each model and separated by space for multiple runs"
)
parser.add_argument(
    '--model-filename', type=str, default='best_valid.pt', help="The models' filenames"
)
parser.add_argument(
    '--data-path', type=str, default='data', help="The path containing the data"
)
parser.add_argument(
    '--device', type=str, default='cuda', help="The device id"
)
parser.add_argument(
    '--batch-size', type=int, default=100, help="The batch size"
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    device = torch.device(args.device)

    # Load data
    ds = Dataset(args.dataset, device, reciprocal=False, data_path=args.data_path)
    n_entities, n_relations, _ = ds.get_shape()

    # Set some useful constants
    batch_size = args.batch_size
    models = args.models.split()
    exp_ids = args.exp_ids.split()
    if len(exp_ids) == 1:
        exp_ids = exp_ids * len(models)
    run_names = args.run_names.split(';')
    run_names = list(map(lambda x: x.split(), run_names))
    assert len(run_names) == len(models), "You should specify one or more runs for each model"

    nrows, ncols = 1, 1
    setup_tueplots(nrows, ncols, rel_width=0.43, hw_ratio=0.67)
    fig, ax = plt.subplots(nrows, ncols)

    # Retrieve run settings
    run_settings = [list(map(lambda x: (int(x.split('_')[-1][1:]), x), rns)) for rns in run_names]
    run_settings = [sorted(rns, key=lambda x: x[0]) for rns in run_settings]
    markers = ['o', 's', '^', 'D', 'h', '*']
    for idx, (m, e, rs) in enumerate(zip(models, exp_ids, run_settings)):
        x_ranks, y_mrrs = list(), list()
        for rank, r in rs:
            # Load the model
            model = load_model(
                args.models_path, args.dataset, n_entities, n_relations,
                m, e, r, args.model_filename, device=device
            )

            # Evaluate the model
            (mrrs, hits, _, _), diverged = ds.eval(model, 'test')
            assert not diverged, f"{m}: Run {r} Diverged"
            x_ranks.append(rank)
            y_mrrs.append(average_metrics_entity(mrrs, hits)['mrr'])
        ax.plot(x_ranks, y_mrrs, markersize=5, c=f'C{idx + 1 if idx > 0 else idx}',
                marker=markers[idx], label=format_model_name(m), alpha=0.7)

    ax.set_xscale('log')
    ax.grid(linestyle='--', alpha=0.3, linewidth=.5)
    ax.legend()
    ax.set_ylabel('Test MRR')
    ax.set_xlabel('Embedding size')

    filename = "rank-ablation-{}".format('-'.join(list(map(str.lower, models))))
    plt.savefig(f'{filename}.pdf')
