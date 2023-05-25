from typing import Tuple

import numpy as np

from sklearn.decomposition import NMF
from tqdm import tqdm


def nnmf_features(
        train_triples: np.ndarray,
        n_entities: int,
        n_relations: int,
        n_components: int = 128
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute NNMF features, using the training triples.
    See "A Probabilistic Framework for Knowledge Graph Data Augmentation"
        @ https://arxiv.org/abs/2110.13205

    :param train_triples: The training triples.
    :param n_entities: The number of entities.
    :param n_relations: The number of relation types.
    :param n_components: The number of components for the Non-Negative Matrix decomposition.
    :return: The head-relation and tail-relation matrices, and the NNMF decomposition of the co-occurrence matrix.
    """
    assert n_components > 0
    print("[NNMF] Computing the co-occurence matrix...")
    hr_mat, tr_mat = compute_hr_tr_matrices(train_triples, n_entities, n_relations)
    occ_mat = hr_mat.dot(tr_mat.T)
    print("[NNMF] Decomposing the co-occurence matrix...")
    transformer = NMF(n_components, max_iter=1000, init='random', random_state=42)
    nnmf_mat1 = transformer.fit_transform(occ_mat)
    nnmf_mat2 = transformer.components_
    return hr_mat, tr_mat, (nnmf_mat1, nnmf_mat2)


def compute_hr_tr_matrices(triples: np.ndarray, n_entities: int, n_relations: int) -> Tuple[np.ndarray, np.ndarray]:
    hr_mat = np.zeros([n_entities, n_relations], dtype=np.float32)
    tr_mat = np.zeros([n_entities, n_relations], dtype=np.float32)
    for i in range(len(triples)):
        s, p, o = triples[i].tolist()
        hr_mat[s, p] += 1
        tr_mat[o, p] += 1
    return hr_mat, tr_mat


def construct_negative_triples(
        n_entities: int,
        n_relations: int,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        test_data: np.ndarray,
        method: str = 'random',
        use_valid: bool = False,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    random_state = np.random.RandomState(seed)
    target_data = valid_data if use_valid else test_data
    if target_data.shape[1] > 3 and method == 'constraint-n':
        num_neg = (target_data.shape[1] - 5) // 2
        target_idx = np.arange(len(target_data))
        obj_neg = target_data[target_idx, 3 + num_neg + random_state.choice(num_neg, size=len(target_data))]
        target_data[:, 2] = obj_neg
        return target_data[:, :3], target_idx
    all_data = np.concatenate([train_data, valid_data, test_data])
    neg_triples = list()
    test_indices = list()
    if method == 'random':
        for i in tqdm(range(len(target_data))):
            (s, p, _) = target_data[i]
            coll_data = all_data[(all_data[:, 0] == s) & (all_data[:, 1] == p)]
            candidate_neg_os = random_state.permutation(n_entities)
            for neg_o in candidate_neg_os:
                if np.any(coll_data[:, 2] == neg_o):
                    continue
                neg_triples.append((s, p, neg_o))
                test_indices.append(i)
                break
    elif method == 'constraint-n':
        for i in tqdm(range(len(target_data))):
            (s, p, _) = target_data[i]
            coll_data = all_data[(all_data[:, 0] == s) & (all_data[:, 1] == p)]
            mask = train_data[:, 1] == p
            candidate_neg_os = random_state.permutation(train_data[mask, 2])
            for neg_o in candidate_neg_os:
                if np.any(coll_data[:, 2] == neg_o):
                    continue
                neg_triples.append((s, p, neg_o))
                test_indices.append(i)
                break
    else:
        raise ValueError("Unknown way of constructing negative triples")
    neg_triples = np.array(neg_triples, dtype=test_data.dtype)
    test_indices = np.array(test_indices, dtype=np.int64)
    return neg_triples, test_indices
