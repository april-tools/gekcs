import pytest
import torch
import numpy as np

from test_utils import generate_all_triples

from kbc.gekc_models import TypedSquaredCP, TypedSquaredComplEx

num_entities, num_relations = 20, 5

pred_domains = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
dom_to_types = {0: (2, 0), 1: (0, 2), 2: (1, 1)}
dom_to_preds = {0: [0, 1], 1: [2], 2: [3, 4]}
type_entity_ids = {0: (0, 1), 1: (1, 9), 2: (9, 20)}


def generate_consistent_triples(num_entities, num_relations):
    triples = generate_all_triples(num_entities, num_relations)
    masked_triples = list()
    for t in triples:
        is_valid = False
        for d in dom_to_types.keys():
            s_type, o_type = dom_to_types[d]
            s_range, o_range = type_entity_ids[s_type], type_entity_ids[o_type]
            preds = dom_to_preds[d]
            if (t[0] >= s_range[0]) and (t[0] < s_range[1]):
                if t[1] in preds:
                    if (t[2] >= o_range[0]) and (t[2] < o_range[1]):
                        is_valid = True
                        break
        if is_valid:
            masked_triples.append(t)
    return np.asarray(masked_triples)


def check_evi_probdist(model, triples):
    # Probabilities of all triples should sum up to 1
    log_probs, _ = model.log_likelihood(triples)

    assert log_probs.shape == (len(triples), 1)
    assert torch.allclose(torch.logsumexp(log_probs, dim=0).exp(), torch.tensor(1.0))


def check_con_probdist(model, triples):
    # Conditional probabilities should sum up to 1 for all the targets
    _, (rhs_log_probs, rel_log_probs, lhs_log_probs) = \
        model.log_likelihood(triples, con_rhs=True, con_rel=True, con_lhs=True, return_ll=False)

    for d in dom_to_types.keys():
        preds = dom_to_preds[d]
        s_type, o_type = dom_to_types[d]
        s_range, o_range = type_entity_ids[s_type], type_entity_ids[o_type]
        s_d, r_d, o_d = s_range[0], preds[0], o_range[0]
        mask = torch.isin(triples[:, 1], torch.tensor(preds))
        m_rhs_con_probs = rhs_log_probs[mask & torch.logical_and(triples[:, 0] == s_d, triples[:, 1] == r_d)]
        m_rel_con_probs = rel_log_probs[mask & torch.logical_and(triples[:, 0] == s_d, triples[:, 2] == o_d)]
        m_lhs_con_probs = lhs_log_probs[mask & torch.logical_and(triples[:, 1] == r_d, triples[:, 2] == o_d)]
        assert torch.allclose(torch.logsumexp(m_rhs_con_probs, dim=0).exp(), torch.tensor(1.0))
        assert torch.allclose(torch.logsumexp(m_rel_con_probs, dim=0).exp(), torch.tensor(1.0))
        assert torch.allclose(torch.logsumexp(m_lhs_con_probs, dim=0).exp(), torch.tensor(1.0))


@pytest.mark.parametrize("rank", [1, 16])
def test_typed_squared_cp(rank):
    triples = torch.LongTensor(generate_consistent_triples(num_entities, num_relations))
    model = TypedSquaredCP((num_entities, num_relations, num_entities), rank)
    model.set_type_constraint_info(pred_domains, dom_to_types, dom_to_preds, type_entity_ids)
    check_con_probdist(model, triples)
    check_evi_probdist(model, triples)


@pytest.mark.parametrize("rank", [1, 16])
def test_typed_squared_complex(rank):
    triples = torch.LongTensor(generate_consistent_triples(num_entities, num_relations))
    model = TypedSquaredComplEx((num_entities, num_relations, num_entities), rank)
    model.set_type_constraint_info(pred_domains, dom_to_types, dom_to_preds, type_entity_ids)
    check_con_probdist(model, triples)
    check_evi_probdist(model, triples)
