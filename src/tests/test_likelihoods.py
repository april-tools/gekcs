import pytest
import torch

from test_utils import generate_all_triples, generate_all_queries

from kbc.gekc_models import NNegCP, NNegComplEx
from kbc.gekc_models import SquaredCP, SquaredComplEx

num_entities, num_relations = 50, 10


def check_evi_probdist(model, triples):
    # Pobabilities of all triples should sum up to 1
    log_probs, _ = model.log_likelihood(triples)

    assert log_probs.shape == (len(triples), 1)
    assert torch.allclose(torch.logsumexp(log_probs, dim=0).exp(), torch.tensor(1.0))


def check_con_probdist(model, triples):
    # Conditional probabilities should sum up to 1 for all the targets
    _, (rhs_log_probs, rel_log_probs, lhs_log_probs) = \
        model.log_likelihood(triples, con_rhs=True, con_rel=True, con_lhs=True, return_ll=False)
    rhs_con_probs = rhs_log_probs[torch.logical_and(triples[:, 0] == 0, triples[:, 1] == 0)]
    rel_con_probs = rel_log_probs[torch.logical_and(triples[:, 0] == 0, triples[:, 2] == 0)]
    lhs_con_probs = lhs_log_probs[torch.logical_and(triples[:, 1] == 0, triples[:, 2] == 0)]

    assert rhs_con_probs.shape == (model.sizes[2], 1)
    assert rel_con_probs.shape == (model.sizes[1], 1)
    assert lhs_con_probs.shape == (model.sizes[0], 1)
    assert torch.allclose(torch.logsumexp(rhs_con_probs, dim=0).exp(), torch.tensor(1.0))
    assert torch.allclose(torch.logsumexp(rel_con_probs, dim=0).exp(), torch.tensor(1.0))
    assert torch.allclose(torch.logsumexp(lhs_con_probs, dim=0).exp(), torch.tensor(1.0))


def check_eval_circuit(model, triples):
    scores = model.score(triples)
    assert scores.shape == (len(triples), 1)

    rhs_cands = model.get_candidates(0, model.sizes[2], target='rhs')
    rel_cands = model.get_candidates(0, model.sizes[1], target='rel')
    lhs_cands = model.get_candidates(0, model.sizes[0], target='lhs')

    rhs_queries, rel_queries, lhs_queries = generate_all_queries(triples.tolist())
    rhs_queries = model.get_queries(torch.LongTensor(rhs_queries), target='rhs')
    rel_queries = model.get_queries(torch.LongTensor(rel_queries), target='rel')
    lhs_queries = model.get_queries(torch.LongTensor(lhs_queries), target='lhs')

    rhs_scores = model.eval_circuit_all(rhs_queries, rhs_cands)
    rel_scores = model.eval_circuit_all(rel_queries, rel_cands)
    lhs_scores = model.eval_circuit_all(lhs_queries, lhs_cands)

    assert rhs_scores.shape == (model.sizes[0] * model.sizes[1], model.sizes[2])
    assert rel_scores.shape == (model.sizes[0] * model.sizes[2], model.sizes[1])
    assert lhs_scores.shape == (model.sizes[1] * model.sizes[2], model.sizes[0])

    assert torch.all(~torch.isnan(rhs_scores)) and torch.all(~torch.isinf(rhs_scores))
    assert torch.all(~torch.isnan(rel_scores)) and torch.all(~torch.isinf(rel_scores))
    assert torch.all(~torch.isnan(lhs_scores)) and torch.all(~torch.isinf(lhs_scores))

    rel_scores = rel_scores.view(model.sizes[0], model.sizes[2], model.sizes[1])
    rel_scores = torch.transpose(rel_scores, 1, 2).contiguous()
    lhs_scores = torch.transpose(lhs_scores, 0, 1).contiguous()

    assert torch.allclose(rhs_scores.view(-1, 1), scores, atol=5e-7)
    assert torch.allclose(rel_scores.view(-1, 1), scores, atol=5e-7)
    assert torch.allclose(lhs_scores.view(-1, 1), scores, atol=5e-7)


@pytest.mark.parametrize("rank", [1, 16])
def test_nneg_cp(rank):
    triples = torch.LongTensor(generate_all_triples(num_entities, num_relations))
    model = NNegCP((num_entities, num_relations, num_entities), rank)
    check_evi_probdist(model, triples)
    check_con_probdist(model, triples)
    check_eval_circuit(model, triples)


@pytest.mark.parametrize("rank", [1, 16])
def test_nneg_complex(rank):
    triples = torch.LongTensor(generate_all_triples(num_entities, num_relations))
    model = NNegComplEx((num_entities, num_relations, num_entities), rank)
    check_evi_probdist(model, triples)
    check_con_probdist(model, triples)
    check_eval_circuit(model, triples)


@pytest.mark.parametrize("rank", [1, 16])
def test_squared_cp(rank):
    triples = torch.LongTensor(generate_all_triples(num_entities, num_relations))
    model = SquaredCP((num_entities, num_relations, num_entities), rank)
    check_evi_probdist(model, triples)
    check_con_probdist(model, triples)
    check_eval_circuit(model, triples)


@pytest.mark.parametrize("rank", [1, 16])
def test_squared_complex(rank):
    triples = torch.LongTensor(generate_all_triples(num_entities, num_relations))
    model = SquaredComplEx((num_entities, num_relations, num_entities), rank)
    check_evi_probdist(model, triples)
    check_con_probdist(model, triples)
    check_eval_circuit(model, triples)
