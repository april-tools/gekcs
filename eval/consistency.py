import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.utils import setup_data_loader, load_model, retrieve_best_hparams, build_run_id
from kbc.datasets import TypedDataset
from kbc.models import KBCModel, MODELS_NAMES
from kbc.gekc_models import TSRL_MODELS_NAMES, TractableKBCModel, TypedSquaredKBCModel
from kbc.utils import set_seed


def predict_inconsistent_hits_at(
        model: KBCModel,
        dataset: TypedDataset,
        data_loader: DataLoader,
        at_k: List[int]
) -> Tuple[Dict[int, float], Dict[int, float], Tuple[np.ndarray, np.ndarray]]:
    domain_entity_ranges = list()
    for d_id in range(len(dataset.type_entity_ids)):
        a, b = dataset.type_entity_ids[d_id]
        domain_entity_ranges.append((a, b))
    domain_entity_ranges = torch.tensor(domain_entity_ranges, device=device)
    rel_to_obj_domain = list()
    rel_to_subj_domain = list()
    for r in range(n_relations):
        s_d_id, o_d_id = dataset.pred_to_types[r]
        rel_to_obj_domain.append(o_d_id)
        rel_to_subj_domain.append(s_d_id)
    rel_to_obj_domain = torch.tensor(rel_to_obj_domain, device=device)
    rel_to_subj_domain = torch.tensor(rel_to_subj_domain, device=device)
    inconsistent_hits = {
        'rhs': defaultdict(lambda: torch.tensor([], device=device)),
        'lhs': defaultdict(lambda: torch.tensor([], device=device))
    }
    consistent_sems = {
        'rhs': defaultdict(lambda: torch.tensor([], device=device)),
        'lhs': defaultdict(lambda: torch.tensor([], device=device))
    }
    inconsistencies_at1 = list()
    inconsistencies_at1_scores = list()
    for (batch,) in tqdm(data_loader):
        # Select subject, relation type, object, and type ids of subject and object
        batch = batch[:, :3].to(device)
        for target in ['rhs', 'lhs']:
            rel_to_ent_domain = rel_to_obj_domain if target == 'rhs' else rel_to_subj_domain
            qs = model.get_queries(batch, target=target)
            if isinstance(model, TractableKBCModel):
                cs = model.get_candidates(0, n_entities, target=target, device=device)
            else:
                cs = model.get_candidates(0, n_entities, target=target)
            if isinstance(model, TractableKBCModel):
                scores = model.eval_circuit_all(qs, cs)  # (b, n)
            else:
                scores = qs @ cs  # (b, n)
            entity_ranges = domain_entity_ranges[rel_to_ent_domain[batch[:, 1]]]  # (b, 2)
            consistency_mask = torch.arange(scores.shape[1]).repeat(scores.shape[0], 1).to(device)  # (b, n)  b -> ith=i
            consistency_mask = (consistency_mask >= entity_ranges[:, [0]]) & (consistency_mask < entity_ranges[:, [1]])
            assert torch.all(torch.any(consistency_mask, dim=1))
            sorted_scores, permutation = torch.sort(scores, descending=True, dim=1)
            sorted_inconsistency_mask = torch.gather(~consistency_mask, dim=1, index=permutation)
            sorted_inconsistency_binary = sorted_inconsistency_mask.long()
            for k in at_k:
                batch_hits_k = (torch.sum(sorted_inconsistency_binary[:, :k], dim=1) > 0).float()
                batch_sem_k = (k - torch.sum(sorted_inconsistency_binary[:, :k], dim=1)).float() / k
                inconsistent_hits[target][k] = torch.concat([inconsistent_hits[target][k], batch_hits_k], dim=0)
                consistent_sems[target][k] = torch.concat([consistent_sems[target][k], batch_sem_k], dim=0)
                if k > 1:
                    continue
                mask = sorted_inconsistency_mask[:, 0]
                if not torch.any(mask):  # at least for one triple in the batch there is an inconsistent hit@1 answer
                    continue
                true_triples = batch[mask]
                inconsistent_ents = permutation[mask, 0].unsqueeze(dim=1)
                if target == 'rhs':
                    inconsistent_rest, true_ents = batch[mask, :2], batch[mask, 2]
                    inconsistent_triples = torch.concat([inconsistent_rest, inconsistent_ents], dim=1)
                else:
                    inconsistent_rest, true_ents = batch[mask, 1:], batch[mask, 0]
                    inconsistent_triples = torch.concat([inconsistent_ents, inconsistent_rest], dim=1)
                true_triples, inconsistent_triples = true_triples.cpu().numpy(), inconsistent_triples.cpu().numpy()
                inconsistencies_at1.append(np.stack([inconsistent_triples, true_triples], axis=1))
                inconsistent_scores = sorted_scores[mask, 0].cpu().numpy()
                filtered_scores = scores[mask]
                filtered_idx = torch.arange(len(filtered_scores), device=filtered_scores.device)
                true_scores = filtered_scores[filtered_idx, true_ents].cpu().numpy()
                inconsistencies_at1_scores.append(np.stack([inconsistent_scores, true_scores], axis=1))

    hits_at, sems_at = dict(), dict()
    for target in ['rhs', 'lhs']:
        hits_at[target] = dict(map(lambda x: (x[0], torch.mean(x[1]).item()), inconsistent_hits[target].items()))
        sems_at[target] = dict(map(lambda x: (x[0], torch.mean(x[1]).item()), consistent_sems[target].items()))
    hits_at = dict(map(lambda k: (k, 0.5 * (hits_at['rhs'][k] + hits_at['lhs'][k])), hits_at['rhs'].keys()))
    sems_at = dict(map(lambda k: (k, 0.5 * (sems_at['rhs'][k] + sems_at['lhs'][k])), sems_at['rhs'].keys()))

    inconsistencies_at1 = np.concatenate(inconsistencies_at1, axis=0) if inconsistencies_at1 else np.array([])
    inconsistencies_at1_scores = np.concatenate(inconsistencies_at1_scores, axis=0) if inconsistencies_at1_scores else np.array([])
    return hits_at, sems_at, (inconsistencies_at1, inconsistencies_at1_scores)


parser = argparse.ArgumentParser(
    description="(ogbl-biokg) Evaluate Predictions Consistency Script"
)
parser.add_argument(
    'models_path', type=str, help="The root path containing the models"
)
parser.add_argument(
    'dataset', type=str, choices=['ogbl-biokg'], help="The evaluation dataset"
)
parser.add_argument(
    'model', type=str, choices=MODELS_NAMES + TSRL_MODELS_NAMES, help="The model to evaluate"
)
parser.add_argument(
    '--exp-id', type=str, default='PLL', help="The model's experiment ID"
)
parser.add_argument(
    '--run-names', required=True, type=str, help="The model's run names, separated by space"
)
parser.add_argument(
    '--model-filename', type=str, default='best_valid.pt', help="The models' filenames"
)
parser.add_argument(
    '--data-path', type=str, default='data', help="The path containing the data"
)
parser.add_argument(
    '--k-values', type=str, default='1 20 100', help="K values for inconsistent hits@k separated by space"
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
    '--show-inconsistencies', action='store_true', help="Whether to print the inconsistent triples found"
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Disable autograd, set seed and device
    torch.autograd.set_grad_enabled(False)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Set some useful constants
    batch_size = args.batch_size
    run_names = args.run_names

    # Load dataset
    ds = TypedDataset(args.dataset, device, reciprocal=False, data_path=args.data_path)
    n_entities, n_relations, _ = ds.get_shape()
    test_data = ds.get_split('test')
    data_loader = setup_data_loader(test_data, batch_size=batch_size)

    # Get the run formats
    run_settings = list(map(lambda r: (int(r.split('_')[-1][1:]), r), run_names))
    run_settings = sorted(run_settings, key=lambda x: x[0])
    for rank, run_name in run_settings:
        # Load the model
        model = load_model(
            args.models_path, args.dataset, n_entities, n_relations,
            args.model, args.exp_id, run_name, args.model_filename, device=device
        )

        # Compute inconsistent hits@k
        hits_at, sems_at, (inconsistencies_at1, inconsitencies_at1_scores) = \
            predict_inconsistent_hits_at(model, ds, data_loader, at_k=list(map(int, args.k_values.split())))
        print(f"{args.model} - {rank}: run={run_name}")
        for k, v in hits_at.items():
            print(f"Inconsistent Hits@{k}: " + "{:.2f}".format(100 * v))
        for k, v in sems_at.items():
            print(f"[base] Sem@{k}: " + "{:.2f}".format(100 * v))
        if args.show_inconsistencies:
            print("Highest scores > True triples scores | Inconsistent @ 1 Triple ; Actual True Triple:")
            print(inconsistencies_at1.shape)
            for i in range(len(inconsistencies_at1)):
                print(f"{inconsitencies_at1_scores[i, 0]} > {inconsitencies_at1_scores[i, 1]}: "
                      f"{inconsistencies_at1[i, 0]} | {inconsistencies_at1[i, 1]}")
        print()
