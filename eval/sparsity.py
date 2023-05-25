import argparse

import matplotlib.pyplot as plt
import torch

from eval.utils import load_model, eval_log_likelihood, setup_tueplots
from kbc.datasets import Dataset
from kbc.models import MODELS_NAMES, ComplEx, CP
from kbc.gekc_models import TSRL_MODELS_NAMES, SquaredKBCModel, TractableKBCModel
from kbc.utils import average_metrics_entity

parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Evaluate Link Prediction Script"
)
parser.add_argument(
    'models_path', type=str, help="The root path containing the models"
)
parser.add_argument(
    'dataset', type=str, help="The evaluation dataset"
)
parser.add_argument(
    'model', type=str, choices=TSRL_MODELS_NAMES + MODELS_NAMES, help="The model name"
)
parser.add_argument(
    '--model-exp-id', type=str, default='1vsAll', help="The experiment ID of the model"
)
parser.add_argument(
    '--model-run', type=str, default="", help="The model's run name, if needed"
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
parser.add_argument(
    '--quantiles', type=str, default=".0 .25 .5 .6 .7 .75 .8 .85 .9 .925 .95 .975 .99",
    help="Quantiles separated by space"
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    device = torch.device(args.device)

    nrows, ncols = 1, 1
    setup_tueplots(nrows, ncols)
    fig, ax = plt.subplots(nrows, ncols)

    # Load data
    ds = Dataset(args.dataset, device, reciprocal=False, data_path=args.data_path)
    n_entities, n_relations, _ = ds.get_shape()

    # Load the model
    model = load_model(
        args.models_path, args.dataset, n_entities, n_relations,
        args.model, args.model_exp_id, args.model_run, args.model_filename,
        device=device, checkpoint_id='gen'
    )
    if isinstance(model, SquaredKBCModel):
        ent_param_name = 'ent_embeddings.weight'
        rel_param_name = 'rel_embeddings.weight'
    elif isinstance(model, CP):
        ent_param_name = 'entity.weight'
        rel_param_name = 'relation.weight'
    elif isinstance(model, ComplEx):
        ent_param_name = 'embeddings.0.weight'
        rel_param_name = 'embeddings.1.weight'
    else:
        raise ValueError("Unsupported model for this evaluating script")

    state_dict = model.state_dict()
    ent_embeddings = state_dict[ent_param_name].clone()
    rel_embeddings = state_dict[rel_param_name].clone()
    quants = torch.tensor(list(map(float, args.quantiles.split())), dtype=torch.float32, device=device)
    abs_ent_quant = torch.quantile(torch.abs(ent_embeddings), q=quants, dim=1)
    abs_rel_quant = torch.quantile(torch.abs(rel_embeddings), q=quants, dim=1)

    # Put q percentage of smaller parameters in magnitude to zero, and evaluate the model
    qs = list()
    qs_mrrs = list()
    qs_mlls = list()
    for i in range(len(quants)):
        mask_ent = torch.abs(ent_embeddings) <= abs_ent_quant[i].unsqueeze(dim=1)
        mask_rel = torch.abs(rel_embeddings) <= abs_rel_quant[i].unsqueeze(dim=1)
        print("q={:.3f}".format(quants[i].item()))
        q_ent_embeddings = ent_embeddings.clone()
        q_rel_embeddings = rel_embeddings.clone()
        q_ent_embeddings[mask_ent] = 0.0
        q_rel_embeddings[mask_rel] = 0.0
        print("Fraction of zero-ed: {:.3f} -- {:.3f}".format(
            1.0 - torch.count_nonzero(q_ent_embeddings) / q_ent_embeddings.numel(),
            1.0 - torch.count_nonzero(q_rel_embeddings) / q_rel_embeddings.numel(),
        ))
        model.load_state_dict({
            ent_param_name: q_ent_embeddings,
            rel_param_name: q_rel_embeddings
        }, strict=False)

        # Evaluate the model
        (mrrs, hits, _, _), diverged = ds.eval(model, 'test')
        assert not diverged, "Diverged"
        metrics = average_metrics_entity(mrrs, hits)
        print(metrics)

        if isinstance(model, TractableKBCModel):
            # Compute the average test log-likelihood too (and one stdev)
            mean_ll, stdev_ll = eval_log_likelihood(
                model, ds.get_examples('test'),
                batch_size=args.batch_size, device=device
            )
            print("[Test] Avg. LL: {:.3f} -- Stddev. LL: {:.3f}".format(mean_ll, stdev_ll))
            qs_mlls.append((mean_ll, stdev_ll))

        qs.append(quants[i].item())
        qs_mrrs.append(metrics['mrr'])

    label_mrr = ax.plot(qs, qs_mrrs, label='Test MRR', color='C0')
    if isinstance(model, TractableKBCModel):
        mll_ax = ax.twinx()
        label_mll = mll_ax.plot(qs, list(map(lambda x: x[0], qs_mlls)), label='Test MLL', color='C1')
        mll_ax.set_ylabel('Test MLL')
        lns = label_mrr + label_mll
    else:
        lns = label_mrr
    labs = [l.get_label() for l in lns]
    ax.set_xscale('log')
    ax.set_xlabel('Fraction of zero-ed parameters')
    ax.set_ylabel('Test MRR')
    ax.set_title(f'{args.dataset} - {args.model} ({args.model_exp_id})')
    ax.legend(lns, labs, loc='lower left')
    plt.savefig('sparsity-metrics-{}-{}-{}.pdf'.format(args.dataset.lower(), args.model.lower(), args.model_exp_id))
