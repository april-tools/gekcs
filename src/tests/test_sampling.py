import pytest
import torch

from kbc.gekc_models import NNegCP, NNegComplEx, SquaredCP, SquaredComplEx

num_entities, num_relations = 50, 10


def check_sampling(model):
    n_samples = 5000
    triples = model.sample(n_samples)
    assert len(triples.shape) == 2
    assert triples.shape[0] == n_samples and triples.shape[1] == 3
    assert torch.all(torch.isin(triples[:, 0], torch.arange(num_entities)))
    assert torch.all(torch.isin(triples[:, 1], torch.arange(num_relations)))
    assert torch.all(torch.isin(triples[:, 2], torch.arange(num_entities)))


@pytest.mark.parametrize("rank", [1, 16])
def test_nneg_cp(rank):
    model = NNegCP((num_entities, num_relations, num_entities), rank)
    check_sampling(model)


@pytest.mark.parametrize("rank", [1, 16])
def test_nneg_complex(rank):
    model = NNegComplEx((num_entities, num_relations, num_entities), rank)
    check_sampling(model)


@pytest.mark.parametrize("rank", [1, 16])
def test_squared_cp(rank):
    model = SquaredCP((num_entities, num_relations, num_entities), rank)
    check_sampling(model)


@pytest.mark.parametrize("rank", [1, 16])
def test_squared_complex(rank):
    model = SquaredComplEx((num_entities, num_relations, num_entities), rank)
    check_sampling(model)
