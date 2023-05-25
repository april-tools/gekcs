import os
import pickle
from typing import Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ogb import linkproppred

from kbc.models import KBCModel
from kbc.gekc_models import TractableKBCModel

DATASETS_NAMES = [
    'NATIONS', 'UMLS', 'KINSHIP', 'FB15K-237', 'WN18RR', 'ogbl-biokg', 'ogbl-wikikg2'
]


def invert(triples: np.array, n_rel: int, stack: bool = True, include_type: bool = False) -> np.ndarray:
    """
    Given triples, return the version containing reciprocal triples, used in training.
    """
    copy = np.copy(triples)
    tmp = np.copy(copy[:, 0])
    copy[:, 0] = copy[:, 2]
    copy[:, 2] = tmp
    copy[:, 1] += n_rel
    if include_type:
        tmp = np.copy(copy[:, -1])
        copy[:, -1] = copy[:, -2]
        copy[:, -2] = tmp
        if triples.shape[1] > 5:
            num_neg = (copy.shape[1] - 5) // 2
            tmp = np.copy(copy[:, 3:3+num_neg])
            assert tmp.shape[1] == num_neg
            copy[:, 3:3+num_neg] = copy[:, 3+num_neg:3+2*num_neg]
            copy[:, 3+num_neg:3+2*num_neg] = tmp
            del tmp
    if stack:
        return np.vstack((triples, copy))
    return copy


class Dataset:
    def __init__(
        self,
        dataset: str,
        device: torch.device,
        reciprocal: bool,
        data_path: str = 'data',
        seed: int = 42
    ):
        self.name = dataset
        self.device = device
        self.reciprocal = reciprocal
        self.root = os.path.join(data_path, self.name)
        self.data = dict()
        self.splits = ['train', 'valid', 'test']
        self.include_type = self.name in ['ogbl-biokg']
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        for f in self.splits:
            p = os.path.join(self.root, f + '.pickle')
            with open(p, 'rb') as in_file:
                self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        if self.name in ['ogbl-wikikg2']:
            self.bsz_vt = 16
        elif self.name in ['WN18RR', 'ogbl-biokg']:
            self.bsz_vt = 512
        elif self.name in ['FB15K-237']:
            self.bsz_vt = 1024
        else:
            self.bsz_vt = 2048
        print(f'{self.name} Dataset Stats: {self.get_shape()}')

        # Compute filtering dictionary, if necessary
        self.to_skip = {'lhs': None, 'rhs': None}
        if self.name not in ['ogbl-biokg', 'ogbl-wikikg2']:
            to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
            for split in self.splits:
                examples = self.data[split]
                for lhs, rel, rhs in examples:
                    to_skip['lhs'][(rhs, rel + self.n_predicates)].add(lhs)  # reciprocals
                    to_skip['rhs'][(lhs, rel)].add(rhs)
            to_skip_final = {'lhs': dict(), 'rhs': dict()}
            for pos, skip in to_skip.items():
                for query, ans in skip.items():
                    to_skip_final[pos][query] = sorted(list(ans))
            self.to_skip = to_skip_final

        n_train = len(self.get_examples('train')) 
        n_valid = len(self.get_examples('valid'))
        n_test = len(self.get_examples('test'))
        print('Train/Valid/Test {}/{}/{}'.format(n_train, n_valid, n_test))
        tot = n_train + n_valid + n_test
        print('Train/Valid/Test {:.3f}/{:.3f}/{:.3f}'.format(
            n_train / tot, n_valid / tot, n_test / tot)
        )

    def load_negatives(self, split: str = 'test') -> Tuple[np.ndarray, np.ndarray]:
        with open(os.path.join(self.root, f'{split}-negatives.pickle'), 'rb') as f:
            data_dict = pickle.load(f)
            test_negatives = data_dict['neg_triples']
            test_negatives_idx = data_dict['test_idx']
        return test_negatives, test_negatives_idx

    def get_shape(self) -> Tuple[int, int, int]:
        n_predicates = self.n_predicates * 2 if self.reciprocal else self.n_predicates
        return self.n_entities, n_predicates, self.n_entities

    def get_examples(self, split: str) -> np.ndarray:
        """
        Return the raw examples.
        """
        return self.data[split].astype(np.int64, copy=True)

    def get_split(self, split: str = 'train') -> np.ndarray:
        """
        Processed split with reciprocal and unified vocabuary.
        """
        data = self.data[split]
        if self.reciprocal:
            assert split != 'test'
            data = self.invert_triples(data, stack=True)
        return data.astype(np.int64, copy=True)

    def invert_triples(self, triples: np.ndarray, stack: bool = False):
        return invert(triples, self.n_predicates, stack=stack, include_type=self.include_type)

    def get_train_loader(self, batch_size: int, num_workers: int = 0, persistent_workers: bool = False) -> DataLoader:
        """
        Construct the data loader.

        :param batch_size: The batch size.
        :param num_workers: The number of workers for multiprocessing.
        :param persistent_workers: Whether to keep workers alive. This will set to False if num_workers = 0.
        :return: The data loader.
        """
        assert num_workers >= 0, "The number of workers must be non-negative"
        if num_workers == 0:
            persistent_workers = False
        training_split = self.get_split('train')
        train_dataset = TensorDataset(torch.from_numpy(training_split))
        loader = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, persistent_workers=persistent_workers
        )
        return loader

    def eval(
        self,
        model: KBCModel,
        split: str,
        n_queries: int = -1,
        query_type: str = 'both',
        at: Tuple[int] = (1, 3, 10),
        eval_ll: bool = False
    ) -> Tuple[Tuple[dict, dict, dict, dict], bool]:
        print('Evaluate the split {}'.format(split))
        examples = self.get_examples(split)
        query_types = ['rhs', 'lhs'] if query_type == 'both' else [query_type]
        res, mean_reciprocal_rank, hits_at = {}, {}, {}
        diverged = False
        if n_queries > 0:
            perm = self.random_state.permutation(len(examples))[:n_queries]
            examples = examples[perm]

        for m in query_types:
            inverted = m == 'lhs' and self.reciprocal
            if inverted:
                candidate_pos = 'rhs'  # after reversing, the candidates to score are at rhs
                q = self.invert_triples(examples)
            else:
                candidate_pos = m
                q = examples
            q = torch.from_numpy(q).to(self.device)
            if 'ogb' in self.name:
                evaluator = linkproppred.Evaluator(name=self.name)
                metrics, div = model.get_metrics_ogb(
                    q, batch_size=self.bsz_vt,
                    query_type=candidate_pos, evaluator=evaluator
                )
                diverged = diverged or div
                mean_reciprocal_rank[m] = metrics['mrr_list']
                hits_at[m] = torch.FloatTensor([metrics['hits@{}_list'.format(k)] for k in at])
                res = None
            else:
                ranks, predicted, div = model.get_ranking(
                    q, self.to_skip[m],
                    batch_size=self.bsz_vt, candidates=candidate_pos
                )
                diverged = diverged or div
                mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
                hits_at[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(), at
                ))))
                res[m] = {
                    'query': q,  # triples to compute rhs raking among all the entities
                    'rank': ranks,
                    'predicted': predicted
                }

        if eval_ll:
            assert isinstance(model, TractableKBCModel)
            if self.reciprocal:
                examples = self.invert_triples(examples, stack=True)
            q = torch.from_numpy(examples).to(self.device)
            if res is None:
                res = defaultdict(dict)
            examples_ll = torch.empty(len(q), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for i in range(0, len(examples_ll), self.bsz_vt):
                    batch = q[i:i + self.bsz_vt]
                    batch_ll, _ = model.log_likelihood(batch, return_ll=True)
                    examples_ll[i:i + self.bsz_vt] = batch_ll.squeeze(dim=1)
            avg_ll = torch.mean(examples_ll).item()
            if not np.isfinite(avg_ll):
                diverged = True
            lls = {'avg_ll': avg_ll}
        else:
            lls = None
        return (mean_reciprocal_rank, hits_at, lls, res), diverged


class TypedDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        device: torch.device,
        reciprocal: bool,
        data_path: str = 'data',
        seed: int = 42
    ):
        if dataset not in ['ogbl-biokg']:
            raise ValueError(f"{dataset} is not a typed knowledge graph")
        super().__init__(dataset, device, reciprocal, data_path=data_path, seed=seed)
        self.type_entity_ids = dict()  # From domain id to a range of consistent entity ids
        with open(os.path.join(self.root, 'dom_id')) as f:
            for line in f:
                _, d, e_id_begin, e_id_end = line.split('\t')
                self.type_entity_ids[int(d)] = (int(e_id_begin), int(e_id_end))
        self.pred_to_types = dict()
        with open(os.path.join(self.root, 'pred_to_types')) as f:
            for line in f:
                r_id, s_d_id, o_d_id = line.split('\t')
                s_d_id, o_d_id = int(s_d_id), int(o_d_id)
                self.pred_to_types[int(r_id)] = (s_d_id, o_d_id)
        if self.reciprocal:
            for r in range(self.n_predicates):
                (s_d_id, o_d_id) = self.pred_to_types[r]
                self.pred_to_types[r + self.n_predicates] = (o_d_id, s_d_id)
        # Construct a map from relation type to subject-object domain id
        subject_object_domain_ids = dict()
        self.pred_to_domains = dict()
        so_domain_idx = 0
        for r, (s_d_id, o_d_id) in self.pred_to_types.items():
            if (s_d_id, o_d_id) not in subject_object_domain_ids:
                subject_object_domain_ids[(s_d_id, o_d_id)] = so_domain_idx
                so_domain_idx += 1
            self.pred_to_domains[r] = subject_object_domain_ids[(s_d_id, o_d_id)]
        # Construct a map from domain ids to subject and object type ids
        self.dom_to_types = dict((v, k) for k, v in subject_object_domain_ids.items())
        # Construct a map from domain ids to relation type ids
        self.dom_to_preds = defaultdict(list)
        for r, d_id in self.pred_to_domains.items():
            self.dom_to_preds[d_id].append(r)
