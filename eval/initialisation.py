import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.utils import setup_data_loader
from kbc.datasets import Dataset
from kbc.engines import setup_model
from kbc.models import MODELS_NAMES
from kbc.gekc_models import TSRL_MODELS_NAMES, SquaredKBCModel
from kbc.utils import set_seed

parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Evaluate Initialisation Methods"
)
parser.add_argument(
    'dataset', type=str, help="The evaluation dataset"
)
parser.add_argument(
    'model', type=str, choices=TSRL_MODELS_NAMES + MODELS_NAMES, help="The model name"
)
parser.add_argument(
    '--rank', type=int, default=1000, help="The rank to use"
)
parser.add_argument(
    '--init-scale', type=float, default=1e-3, help="Initialisation scale"
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
    '--seed', type=int, default=42, help="The seed"
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    device = torch.device(args.device)
    set_seed(args.seed)

    # Load data
    ds = Dataset(args.dataset, device, reciprocal=False, data_path=args.data_path)
    n_entities, n_relations, _ = ds.get_shape()
    train_data = ds.get_split('train')
    valid_data = ds.get_split('valid')
    test_data = ds.get_split('test')
    batch_size = args.batch_size
    data = np.concatenate([train_data, valid_data, test_data], axis=0)
    data = np.stack([
        np.random.randint(n_entities, size=len(data) * 4),
        np.random.randint(n_relations, size=len(data) * 4),
        np.random.randint(n_entities, size=len(data) * 4)
    ], axis=1)
    data_loader = setup_data_loader(data, batch_size=batch_size)

    # Setup the model
    conf = {'model': args.model, 'device': 'cpu', 'rank': args.rank, 'init_scale': args.init_scale}
    model = setup_model((n_entities, n_relations, n_entities), conf)

    # Show parameters' distributions
    for (name, param) in model.named_parameters():
        param = param.cpu().numpy()
        emb_size = param.shape[1]
        q1, q2 = np.quantile(param, q=[0.0001, 0.9999])
        plt.hist(
            param.reshape(-1), label=name, density=True, range=(q1, q2),
            bins=2 * int(np.sqrt(emb_size)), alpha=0.5
        )
    plt.legend()
    plt.show()
    plt.cla()
    plt.clf()

    # Get the histogram of the scores (un-normalised and in log-space) on the training set
    y_scores = torch.zeros(len(data), dtype=torch.float32)
    for i, (batch,) in enumerate(data_loader):
        batch.to(device)
        y_batch = model.score(batch)
        if isinstance(model, SquaredKBCModel):
            y_batch = torch.log(y_batch + model.eps)
        assert torch.all(torch.isfinite(y_batch))
        y_scores[i*batch_size:i*batch_size+batch_size] = y_batch.squeeze(dim=1).cpu()
    y_scores = y_scores.numpy()
    q1, q2 = np.quantile(y_scores, q=[0.0001, 0.9999])
    plt.hist(
        y_scores, label='log phi', density=True, range=(q1, q2),
        bins=4 * int(np.cbrt(len(y_scores))), alpha=0.8
    )
    plt.legend()
    plt.show()
