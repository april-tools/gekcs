import os
import pickle
import argparse
from typing import Optional

import numpy as np
from ogb import linkproppred

from kbc.datasets import DATASETS_NAMES
from kbc.graph import nnmf_features, construct_negative_triples


def preprocess_dataset(path: str, ds_out_path: str):
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        file_path = os.path.join(path, f"{f}.txt")
        with open(file_path, 'r') as to_read:
            for line in to_read.readlines():
                lhs, rel, rhs = line.strip().split('\t')
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    n_relations = len(relations)
    n_entities = len(entities)
    print(f"{n_entities} entities and {n_relations} relations")

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        with open(os.path.join(ds_out_path, f), 'w+') as ff:
            for (x, i) in dic.items():
                ff.write("{}\t{}\n".format(x, i))

    # map train/test/valid with the ids
    triples = {'train': None, 'valid': None, 'test': None}
    for f in files:
        file_path = os.path.join(path, f"{f}.txt")
        with open(file_path, 'r') as to_read:
            examples = list()
            for line in to_read.readlines():
                lhs, rel, rhs = line.strip().split('\t')
                try:
                    examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
                except ValueError:
                    continue
        with open(os.path.join(ds_out_path, f + '.pickle'), 'wb') as to_write:
            examples_array = np.array(examples).astype(np.int64, copy=True)
            pickle.dump(examples_array, to_write)
            triples[f] = examples_array

    return triples, n_entities, n_relations, relations_to_id


def preprocess_ogbl_dataset(name: str, path: str, ds_out_path: str):
    dataset = linkproppred.LinkPropPredDataset(name, root=path)
    split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

    if name == 'ogbl-biokg':
        cur_idx, cur_type_idx, type_dict, entity_dict = 0, 0, {}, {}
        for key in dataset[0]['num_nodes_dict']:
            type_dict[key] = cur_type_idx
            cur_type_idx += 1
            entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
            cur_idx += dataset[0]['num_nodes_dict'][key]

        def index_triples_across_type(triples, entity_dict, type_dict):
            triples['head_type_idx'] = np.zeros_like(triples['head'])
            triples['tail_type_idx'] = np.zeros_like(triples['tail'])
            for i in range(len(triples['head'])):
                h_type = triples['head_type'][i]
                triples['head_type_idx'][i] = type_dict[h_type]
                triples['head'][i] += entity_dict[h_type][0]
                if 'head_neg' in triples:
                    triples['head_neg'][i] += entity_dict[h_type][0]
                t_type = triples['tail_type'][i]
                triples['tail_type_idx'][i] = type_dict[t_type]
                triples['tail'][i] += entity_dict[t_type][0]
                if 'tail_neg' in triples:
                    triples['tail_neg'][i] += entity_dict[t_type][0]
            return triples

        print('Indexing triples across different entity types ...')
        train_triples = index_triples_across_type(train_triples, entity_dict, type_dict)
        valid_triples = index_triples_across_type(valid_triples, entity_dict, type_dict)
        test_triples = index_triples_across_type(test_triples, entity_dict, type_dict)

        other_data = {
            'train': np.concatenate([
                train_triples['head_type_idx'].reshape(-1, 1),
                train_triples['tail_type_idx'].reshape(-1, 1)
            ], axis=1),
            'valid': np.concatenate([
                valid_triples['head_neg'],
                valid_triples['tail_neg'],
                valid_triples['head_type_idx'].reshape(-1, 1),
                valid_triples['tail_type_idx'].reshape(-1, 1)
            ], axis=1),
            'test': np.concatenate([
                test_triples['head_neg'],
                test_triples['tail_neg'],
                test_triples['head_type_idx'].reshape(-1, 1),
                test_triples['tail_type_idx'].reshape(-1, 1)
            ], axis=1)
        }
    elif name == 'ogbl-wikikg2':
        other_data = {
            'train': None,
            'valid': np.concatenate([
                valid_triples['head_neg'],
                valid_triples['tail_neg']
            ], axis=1),
            'test': np.concatenate([
                test_triples['head_neg'],
                test_triples['tail_neg']
            ], axis=1)
        }
    else:
        raise ValueError(f"Unknown OGB dataset {name}")

    n_relations = int(max(train_triples['relation'])) + 1
    if name == 'ogbl-biokg':
        n_entities = sum(dataset[0]['num_nodes_dict'].values())
        assert train_triples['head'].max() <= n_entities
    elif name == 'ogbl-wikikg2':
        n_entities = int(max(np.concatenate([train_triples['head'], train_triples['tail']]))) + 1
    else:
        raise ValueError(f"Unknown OGB dataset {name}")
    print(f"{n_entities} entities and {n_relations} relations")

    train_array = np.concatenate([
        train_triples['head'].reshape(-1, 1),
        train_triples['relation'].reshape(-1, 1),
        train_triples['tail'].reshape(-1, 1)
    ], axis=1).astype(np.int64, copy=True)
    if other_data['train'] is not None:
        train_array = np.concatenate([train_array, other_data['train']], axis=1).astype(np.int64, copy=True)
    valid_array = np.concatenate([
        valid_triples['head'].reshape(-1, 1),
        valid_triples['relation'].reshape(-1, 1),
        valid_triples['tail'].reshape(-1, 1),
        other_data['valid']
    ], axis=1).astype(np.int64, copy=True)
    test_array = np.concatenate([
        test_triples['head'].reshape(-1, 1),
        test_triples['relation'].reshape(-1, 1),
        test_triples['tail'].reshape(-1, 1),
        other_data['test']
    ], axis=1).astype(np.int64, copy=True)

    triples = {'train': train_array, 'valid': valid_array, 'test': test_array}
    for f in ['train', 'valid', 'test']:
        with open(os.path.join(ds_out_path, f + '.pickle'), 'wb') as to_write:
            pickle.dump(triples[f], to_write)

    if name == 'ogbl-biokg':
        # Write domains to IDs relationships, and IDs ranges of entityes in such domains
        with open(os.path.join(ds_out_path, 'dom_id'), 'w') as to_write:
            for k in type_dict:
                to_write.write(f"{k}\t{type_dict[k]}\t{entity_dict[k][0]}\t{entity_dict[k][1]}\n")
        entity_id_to_type_id = dict()
        # Compute and write relationships between predicates and domains of subjects and objects
        for k, v in entity_dict.items():
            for i in range(v[0], v[1]):
                entity_id_to_type_id[i] = type_dict[k]
        all_triples = np.concatenate([train_array[:, :3], valid_array[:, :3], test_array[:, :3]])
        with open(os.path.join(ds_out_path, 'pred_to_types'), 'w') as to_write:
            for p in range(n_relations):
                p_head_tail = all_triples[all_triples[:, 1] == p][:, [0, 2]]
                p_head_tail_types = list(map(lambda x: (entity_id_to_type_id[x[0]], entity_id_to_type_id[x[1]]), p_head_tail.tolist()))
                p_head_tail_types = np.unique(np.asarray(p_head_tail_types), axis=0)
                assert len(p_head_tail_types) == 1, "Only allowing maps between predicates and subject/object types"
                to_write.write(f"{p}\t{p_head_tail_types[0, 0]}\t{p_head_tail_types[0, 1]}\n")

    return triples, n_entities, n_relations


def prepare_dataset(
        name: str,
        path: str,
        out_path: str,
        nnmf: bool = False,
        nnmf_kwargs: Optional[dict] = None,
        save_negatives: bool = False,
        negatives_method: str = 'random',
        skip: bool = False,
        seed: int = 42
):
    """
    KBC dataset preprocessing.
    1) Maps each entity and relation to a unique id
    2) Create a corresponding folder of `cwd/data/dataset`, with mapped train/test/valid files.
    3) Create `to_skip_lhs` & `to_skip_rhs` for filtered metrics
    4) Save the mapping `rel_id` & `ent_id` for analysis.

    :param name: The name of the dataset.
    :param path: A path of a folder containing 3 tab-separated files, `train`, `valid` and `test`.
    :param out_path: The destination path.
    :param nnmf: Whether to compute Non-Negative Matrix Factorization (NNMF) features.
    :param save_negatives: Whether to generate save negative triples from validation/test triples.
    :param negatives_method: The methot used to sample negative triples from test triples.
     For ogbl datasets the negatives in the test set will be sampled randomly.
    :param skip: Whether to skip already preprocessed datasets.
    :param seed: The seed to use.
    """
    ds_out_path = os.path.join(out_path, name)
    if skip and os.path.isdir(ds_out_path):
        raise OSError(f"Dataset {ds_out_path} has already been preprocessed")
    os.makedirs(ds_out_path, exist_ok=True)

    # Preprocess the non-ogbl dataset
    if 'ogbl' not in name:
        triples, n_entities, n_relations, relations_to_id = preprocess_dataset(path, ds_out_path)
    else:
        triples, n_entities, n_relations = preprocess_ogbl_dataset(name, path, ds_out_path)

    # Save negative triples, if required
    if save_negatives:
        print("Saving negative triples...")
        for split in ['valid', 'test']:
            neg_triples, test_indices = construct_negative_triples(
                n_entities, n_relations,
                triples['train'], triples['valid'], triples['test'],
                method=negatives_method, use_valid=split == 'valid', seed=seed
            )
            with open(os.path.join(ds_out_path, f'{split}-negatives.pickle'), 'wb') as f:
                pickle.dump({'neg_triples': neg_triples, 'test_idx': test_indices}, f)

    if nnmf:
        if nnmf_kwargs is None:
            nnmf_kwargs = dict()
        print("Computing NNMF features...")
        train_triples = triples['train']
        if name in ['ogbl-biokg']:
            train_triples = train_triples[:, :3]
        hr_mat, tr_mat, (nnmf_mat1, nnmf_mat2) = nnmf_features(train_triples, n_entities, n_relations, **nnmf_kwargs)
        nnmf_mat = np.concatenate([nnmf_mat1, nnmf_mat2.T], axis=1)
        nnmf_mat = nnmf_mat.astype(np.float32, copy=True)
        np.savez(os.path.join(ds_out_path, 'nnmf-features'), hr_mat=hr_mat, tr_mat=tr_mat, nnmf_mat=nnmf_mat)

    print('Done processing!')


parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Datasets Preprocessor"
)
parser.add_argument(
    'datapath', type=str, help="The directory containing the raw data"
)
parser.add_argument(
    '--datasets', type=str, default=' '.join(DATASETS_NAMES), help="The datasets to preprocess (separated by space)"
)
parser.add_argument(
    '--out-datapath', type=str, help="The directory containing the preprocessed data", default='data'
)
parser.add_argument(
    '--nnmf', action='store_true', help="Whether to compute NNMF features too"
)
parser.add_argument(
    '--n-components', type=int, default=256, help="The number of components for NNMF"
)
parser.add_argument(
    '--save-negatives', action='store_true',
    help="Whether to generate and save negative triples from validation/test triples"
)
parser.add_argument(
    '--negatives-method', default='constraint-n', choices=['random', 'constraint-n'],
    help="The method to generate negative triples. For ogbl datasets they will be chosen from the test data"
)
parser.add_argument(
    '--seed', type=int, help="Seed for random operations", default=42
)
parser.add_argument(
    '--skip', action='store_true', help="Whether to skip already preprocessed datasets"
)


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.isdir(args.datapath):
        raise ValueError("The directory containing the raw data is not valid")
    if not os.path.isdir(args.out_datapath):
        os.makedirs(args.out_datapath)

    # Process each dataset
    for ds in args.datasets.split():
        print(f"Preprocessing dataset {ds}")
        ds_path = os.path.join(args.datapath, ds)
        try:
            prepare_dataset(
                ds,
                ds_path,
                args.out_datapath,
                nnmf=args.nnmf,
                nnmf_kwargs={'n_components': args.n_components},
                save_negatives=args.save_negatives,
                negatives_method=args.negatives_method,
                skip=args.skip,
                seed=args.seed
            )
        except OSError as ex:
            print(ex)
