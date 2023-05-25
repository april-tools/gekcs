import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from eval.utils import load_model, setup_tueplots, format_model_name, eval_log_likelihood
from kbc.datasets import Dataset, DATASETS_NAMES
from kbc.gekc_models import TractableKBCModel
from kbc.utils import set_seed


def sample_independently_uniformly(n_entities: int, n_relations: int, n_samples: int) -> torch.Tensor:
    s = np.random.randint(n_entities, size=n_samples)
    p = np.random.randint(n_relations, size=n_samples)
    o = np.random.randint(n_entities, size=n_samples)
    triples = np.stack([s, p, o], axis=1)
    return torch.from_numpy(triples)


def sample_nnmfaug_method(
        data_path: str,
        n_clusters: int,
        n_samples: int,
        alpha: float = 1e-3
) -> Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with open(os.path.join(data_path, 'nnmf-features.npz'), 'rb') as f:
        features = np.load(f)
        hr_mat, tr_mat, nnmf_mat = features['hr_mat'], features['tr_mat'], features['nnmf_mat']
    clusters = AgglomerativeClustering(n_clusters).fit_predict(nnmf_mat)
    triples = np.empty((n_samples, 3), dtype=np.int64)
    for i in tqdm(range(n_samples), desc='[NNMFAug]'):
        cc = np.random.randint(n_clusters)
        es = np.argwhere(clusters == cc).flatten()
        s = np.random.choice(es)
        o = np.random.choice(es)
        p_dist = hr_mat[s] * tr_mat[o] + alpha
        p_probs = p_dist / np.sum(p_dist)
        p = np.random.choice(n_relations, p=p_probs)
        triples[i, 0] = s
        triples[i, 1] = p
        triples[i, 2] = o
    return torch.from_numpy(triples), (hr_mat, tr_mat, clusters)


def nnmfaug_method_log_likelihood(
        triples: np.ndarray,
        hr_mat: np.ndarray,
        tr_mat: np.ndarray,
        clusters: np.ndarray,
        alpha: float = 1e-3
):
    unique_clusters, counts_clusters = np.unique(clusters, return_counts=True)
    log_prob_cluster = -np.log(len(unique_clusters))
    triples_clusters = clusters[triples[:, 0]]  # or equivalently, clusters[triples[:, 2]]
    cluster_sizes = counts_clusters[triples_clusters]
    log_prob_ht_given_cluster = -2.0 * np.log(cluster_sizes)
    r_scores = hr_mat[triples[:, 0]] * tr_mat[triples[:, 2]] + alpha
    log_probs_r_given_ht = np.log(r_scores / np.sum(r_scores, axis=1, keepdims=True))
    log_probs_r_given_ht = log_probs_r_given_ht[np.arange(len(log_probs_r_given_ht)), triples[:, 1]]
    return log_prob_cluster + log_prob_ht_given_cluster + log_probs_r_given_ht


def sample_gekc_model(
        models_path: str,
        dataset: str,
        name: str,
        exp_id: str,
        run_name: str,
        model_filename: str,
        n_samples: int,
        device: torch.device
) -> Tuple[TractableKBCModel, torch.Tensor]:
    model: TractableKBCModel = load_model(
        models_path, dataset, n_entities, n_relations,
        name, exp_id, run_name, model_filename, checkpoint_id='gen', device=device
    )
    triples = list()
    for i in tqdm(range(0, n_samples, batch_size), desc=f'[{name}]'):
        ss = batch_size if i + batch_size <= n_samples else n_samples - i
        triples.append(model.sample(ss))
    return model, torch.concat(triples)


def compute_features(x: torch.Tensor, ref_model: nn.Module, desc: str) -> torch.Tensor:
    x_features = torch.empty([x.shape[0], 4 * ref_model.rank], dtype=torch.float32, device=device)
    for i in tqdm(range(0, x.shape[0], batch_size), desc=f'[{desc}] Computing features'):
        batch = x[i:i+batch_size].to(device)
        lhs = ref_model.embeddings[0](batch[:, 0])
        rel = ref_model.embeddings[1](batch[:, 1])
        rhs = ref_model.embeddings[0](batch[:, 2])
        lhs = lhs[:, :ref_model.rank], lhs[:, ref_model.rank:]
        rel = rel[:, :ref_model.rank], rel[:, ref_model.rank:]
        rhs = rhs[:, :ref_model.rank], rhs[:, ref_model.rank:]
        x0 = lhs[0] * rel[0] * rhs[0]
        x1 = lhs[1] * rel[1] * rhs[0]
        x2 = lhs[0] * rel[1] * rhs[1]
        x3 = lhs[1] * rel[0] * rhs[1]
        x_features[i:i+batch_size] = torch.concat([x0, x1, x2, x3], dim=1)
    return x_features


def kid_score(
        real_features: torch.Tensor,
        fake_features: torch.Tensor
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Computes the KID score. Core implementation taken from
    TorchMetrics lib (v0.8.2) https://torchmetrics.readthedocs.io/en/v0.8.2/image/kernel_inception_distance.html.
    """
    def poly_kernel(
            f1: torch.Tensor, f2: torch.Tensor,
            degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> torch.Tensor:
        """Adapted from `KID Score`_"""
        if gamma is None:
            gamma = 1.0 / f1.shape[1]
        kernel = (f1 @ f2.T * gamma + coef) ** degree
        return kernel

    def maximum_mean_discrepancy(k_xx: torch.Tensor, k_xy: torch.Tensor, k_yy: torch.Tensor) -> torch.Tensor:
        m = k_xx.shape[0]
        diag_x = torch.diag(k_xx)
        diag_y = torch.diag(k_yy)

        kt_xx_sums = k_xx.sum(dim=-1) - diag_x
        kt_yy_sums = k_yy.sum(dim=-1) - diag_y
        k_xy_sums = k_xy.sum(dim=0)

        kt_xx_sum = kt_xx_sums.sum()
        kt_yy_sum = kt_yy_sums.sum()
        k_xy_sum = k_xy_sums.sum()

        value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
        value -= 2 * k_xy_sum / (m ** 2)
        return value

    def poly_mmd(
            f_real: torch.Tensor, f_fake: torch.Tensor,
            degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
    ) -> torch.Tensor:
        """Adapted from `KID Score`_"""
        k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
        k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
        k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
        return maximum_mean_discrepancy(k_11, k_12, k_22)

    real_features = torch.div(real_features, torch.norm(real_features, p=2, dim=1, keepdim=True))
    fake_features = torch.div(fake_features, torch.norm(fake_features, p=2, dim=1, keepdim=True))

    n_samples_real = real_features.shape[0]
    n_samples_fake = fake_features.shape[0]
    kid_scores_ = []
    for _ in tqdm(range(n_subsets), desc='Computing KID Score', leave=False):
        perm = torch.randperm(n_samples_real)
        f_real = real_features[perm[:subset_size]]
        perm = torch.randperm(n_samples_fake)
        f_fake = fake_features[perm[:subset_size]]
        o = poly_mmd(f_real, f_fake, degree=kappa, gamma=gamma, coef=alpha)
        kid_scores_.append(o)
    kid = torch.stack(kid_scores_).cpu().numpy()
    return kid, (np.mean(kid).item(), np.std(kid).item())


def fit_transform_tsne(x: np.ndarray, n_components: int = 2, perplexity: float = 50.0, n_iter: int = 5000) -> np.ndarray:
    return TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter).fit_transform(x)


parser = argparse.ArgumentParser(
    description="Evaluate Sampling Script"
)
parser.add_argument(
    'models_path', type=str, help="The root path containing the models"
)
parser.add_argument(
    'dataset', type=str, choices=DATASETS_NAMES, help="The evaluation dataset"
)
parser.add_argument(
    '--ref-model', type=str, default='ComplEx', choices=['CP', 'ComplEx'], help="The reference model name"
)
parser.add_argument(
    '--ref-exp-id', required=True, type=str, default='PLL', help="The reference model's experiment ID"
)
parser.add_argument(
    '--ref-run', required=True, type=str, default="", help="The reference model's run name"
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
    '--n-samples', type=int, default=25_000, help="The number of samples"
)
parser.add_argument(
    '--kid-subsets', type=int, default=100, help="The number of subsets for computing the KID score"
)
parser.add_argument(
    '--kid-size', type=int, default=1000, help="The sample size of each subset for computing the KID score"
)
parser.add_argument(
    '--gamma', type=float, default=1.0, help="Polynomial kernel's gamma parameter"
)
parser.add_argument(
    '--alpha', type=float, default=1.0, help="Polynomial kernel's alpha parameter"
)
parser.add_argument(
    '--kappa', type=int, default=3, help="Polynomial kernel's kappa parameter"
)
parser.add_argument(
    '--eval-uniform', action='store_true', help="Whether to evaluate the uniform baseline"
)
parser.add_argument(
    '--eval-nnmfaug', action='store_true', help="Whether to evaluate the NNMFAug baseline"
)
parser.add_argument(
    '--nnmfaug-clusters', type=str, default="50", help="The number of clusters for the NNMFAug framework baseline"
)
parser.add_argument(
    '--eval-gekc-models', type=str, default="", help="List of GeKC models to evaluate, separated by space"
)
parser.add_argument(
    '--gekc-exp-ids', type=str, default="", help="List of experiments IDs of GeKC models, separated by space"
)
parser.add_argument(
    '--gekc-runs', type=str, default="", help="The GeKC models' run names, separated by space"
)
parser.add_argument(
    '--device', type=str, default='cuda', help="The device id"
)
parser.add_argument(
    '--batch-size', type=int, default=500, help="The batch size"
)
parser.add_argument(
    '--plot-embeddings', action='store_true', help="Whether to project triple embeddings into a 2D space and plot it"
)
parser.add_argument(
    '--plot-embeddings-train', action='store_true', help="Whether to project training triple embeddings too"
)
parser.add_argument(
    '--manifold-alpha', type=float, default=0.05, help="The alpha transparency value for the manifold scatter plot"
)
parser.add_argument(
    '--show-legend', action='store_true', help="Whether to show legends"
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Set some useful constants
    n_samples = args.n_samples
    n_subsets = args.kid_subsets
    subset_size = args.kid_size
    gamma = args.gamma
    alpha = args.alpha
    kappa = args.kappa
    batch_size = args.batch_size

    # Load dataset
    ds = Dataset(args.dataset, device, reciprocal=False, data_path=args.data_path)
    n_entities, n_relations, _ = ds.get_shape()
    train_data = ds.get_split('train')
    complete_test_data = ds.get_split('test')
    test_data = complete_test_data
    if args.plot_embeddings and n_samples > len(test_data):
        n_samples = len(test_data)
    if len(train_data) > n_samples:
        train_data = train_data[np.random.choice(len(train_data), size=n_samples, replace=False)]
    if len(test_data) > n_samples:
        test_data = test_data[np.random.choice(len(test_data), size=n_samples, replace=False)]
    train_data = torch.from_numpy(train_data).to(device)
    test_data = torch.from_numpy(test_data).to(device)

    nrows, ncols = 1, 1
    setup_tueplots(nrows, ncols, rel_width=0.66, hw_ratio=1.0)
    fig, ax = plt.subplots(nrows, ncols)

    # Load the reference model
    ref_model = load_model(
        args.models_path, args.dataset, n_entities, n_relations,
        args.ref_model, args.ref_exp_id, args.ref_run, args.model_filename,
        filter_inverted_relations=True, device=device
    )

    # Compute the train, validation and test features
    train_features = compute_features(train_data, ref_model, desc='Train')
    test_features = compute_features(test_data, ref_model, desc='Test')

    # Perform sanity check
    _, (mu_kid, std_kid) = kid_score(train_features, train_features)
    print("[Train/Train] Sanity Check (KID): {:.3f}+-{:.3f}".format(mu_kid, std_kid))
    test_train_kid_scores, (mu_kid, std_kid) = kid_score(test_features, train_features)
    print("[Train/Test] Sanity Check (KID): {:.3f}+-{:.3f}".format(mu_kid, std_kid))
    ref_features = test_features

    proj_features = dict()
    if args.plot_embeddings:
        proj_features['Test'] = ref_features.cpu().numpy()
        if args.plot_embeddings_train:
            proj_features['Training'] = train_features.cpu().numpy()

    # Uniform baseline
    if args.eval_uniform:
        uniform_triples = sample_independently_uniformly(n_entities, n_relations, n_samples)
        uniform_features = compute_features(uniform_triples, ref_model, 'Uniform')
        uniform_kid_scores, (mu_kid, std_kid) = kid_score(ref_features, uniform_features)
        print("[Uniform] (KID): {:.3f}+-{:.3f}".format(mu_kid, std_kid))
        print("[Uniform] (LL): {:.3f}".format(-np.log(n_entities * n_relations * n_entities)))

    # NNMFAug baseline
    if args.eval_nnmfaug:
        data_path = os.path.join(args.data_path, args.dataset)
        for n_clusters in map(int, args.nnmfaug_clusters.split()):
            nnmfaug_triples, (hr_mat, tr_mat, clusters) = sample_nnmfaug_method(data_path, n_clusters, n_samples)
            nnmfaug_features = compute_features(nnmfaug_triples, ref_model, 'NNMFAug')
            nnmfaug_kid_scores, (mu_kid, std_kid) = kid_score(ref_features, nnmfaug_features)
            print("[NNMFAug] [{}] (KID): {:.3f}+-{:.3f}".format(n_clusters, mu_kid, std_kid))
            nnmfaug_ll = nnmfaug_method_log_likelihood(complete_test_data, hr_mat, tr_mat, clusters)
            print("[NNMFAug] [{}] (LL): {:.3f}".format(n_clusters, np.mean(nnmfaug_ll)))

    # GeKC models
    gekc_models = args.eval_gekc_models.split()
    if gekc_models:
        gekc_exp_ids = args.gekc_exp_ids.split()
        gekc_runs = args.gekc_runs.split()
        assert len(gekc_exp_ids) == 1 or len(gekc_exp_ids) == len(gekc_models), "You should specify one experiment ID for each GeKC model, or only one"
        assert len(gekc_models) == len(gekc_runs), "You should specify a run name for each GeKC model"
        if len(gekc_exp_ids) == 1:
            gekc_exp_ids = gekc_exp_ids * len(gekc_models)
        for m, e, r in zip(gekc_models, gekc_exp_ids, gekc_runs):
            model, gekc_triples = sample_gekc_model(
                args.models_path, args.dataset, m, e, r, args.model_filename,
                n_samples=n_samples, device=device
            )
            gekc_features = compute_features(gekc_triples, ref_model, m)
            gekc_kid_scores, (mu_kid, std_kid) = kid_score(ref_features, gekc_features)
            print("[{}] [{}] (KID): {:.3f}+-{:.3f}".format(m, e, mu_kid, std_kid))
            avg_ll, _ = eval_log_likelihood(model, complete_test_data, batch_size=batch_size, device=device)
            print("[{}] [{}] (LL): {:.3f}".format(m, e, avg_ll))
            if args.plot_embeddings:
                samples_id = format_model_name(m)
                proj_features[samples_id] = gekc_features.cpu().numpy()

    if args.plot_embeddings:
        intervals = dict()
        all_features = list()
        base_idx = 0
        proj_keys = list(proj_features.keys())
        for k in proj_keys:
            fs = proj_features[k]
            intervals[k] = (base_idx, base_idx + len(fs))
            base_idx += len(fs)
            all_features.append(fs)
        all_features = np.concatenate(all_features)
        print("Fitting TSNE ...")
        tsne_features = fit_transform_tsne(all_features)
        for k in proj_keys:
            begin_idx, end_idx = intervals[k]
            x = tsne_features[begin_idx:end_idx]
            ax.scatter(x[:, 0], x[:, 1], alpha=args.manifold_alpha, s=1, marker='.', label=k)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        if args.show_legend:
            lgnd = ax.legend(loc='upper left', markerscale=4)
            for i in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[i].set_alpha(1.0)
        plt.savefig(f'{args.dataset.lower()}-triple-embeddings.png', dpi=300)
