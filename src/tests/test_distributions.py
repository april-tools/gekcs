import pytest
import torch

from kbc.distributions import Categorical, TwinCategorical


@pytest.mark.parametrize("num_categories, batch_size", [(100, 1), (100, 32)])
def test_categorical(num_categories, batch_size):
    dist = Categorical(num_categories, batch_size)
    x = torch.arange(num_categories)
    log_probs = dist.forward(x) - dist.partition_function()
    assert log_probs.shape == (num_categories, batch_size)
    assert torch.allclose(torch.logsumexp(log_probs, dim=0).exp(), torch.tensor(1.0))


@pytest.mark.parametrize("num_categories, batch_size", [
    (100, 1), (100, 32)
])
def test_twin_categorical(num_categories, batch_size):
    dist = TwinCategorical(num_categories, batch_size)
    x = torch.arange(num_categories)
    log_scores = dist.forward(x)
    log_probs = log_scores - dist.partition_function()
    assert log_scores.shape == (num_categories, batch_size, 2)
    assert torch.all(log_scores[:, :, 0] >= log_scores[:, :, 1])
    assert torch.allclose(torch.logsumexp(log_probs, dim=0).exp(), torch.tensor(1.0))
