import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Dirichlet


def init_params_(tensor: torch.Tensor, init_dist: str, init_loc: float = 0.0, init_scale: float = 1.0):
    """
    Initialize parameters in-place.

    :param tensor: The tensor to initialize in-place.
    :param init_dist: The initializing distribution.
    :param init_loc: The initializer's shifting hyperparameter.
    :param init_scale: The initializer's scaling hyperparameter.
    """
    if init_dist == 'uniform':
        nn.init.uniform_(tensor, -init_scale, init_scale)
        tensor += init_loc
    elif init_dist == 'normal':
        nn.init.normal_(tensor, init_loc, init_scale)
    elif init_dist == 'xavier-uniform':
        nn.init.xavier_uniform_(tensor, init_scale)
        tensor += init_loc
    elif init_dist == 'log-normal':
        nn.init.normal_(tensor, init_loc, init_scale)
        tensor.exp_()
    elif init_dist == 'exp-dirichlet':
        concentration = torch.full((tensor.shape[0],), fill_value=1.0 / init_scale, dtype=torch.float64)
        t = Dirichlet(concentration).sample(tensor.shape[1:])
        t = t.permute(dims=[len(tensor.shape) - 1] + list(range(len(tensor.shape) - 1)))
        t = torch.log(t)
        tensor.copy_(t.float())
    elif init_dist == 'centered-cp-normal':
        # The initial outputs of CP+ will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc
        tensor.copy_(t.float())
    elif init_dist == 'centered-complex-normal':
        # The initial outputs of ComplEx+ will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(2 * tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc
        tensor.copy_(t.float())
    elif init_dist == 'centered-rescal-normal':
        # The initial outputs of RESCAL+ will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 2.0 + 0.5 * (init_scale ** 2)
        t = torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc
        tensor.copy_(t.float())
    elif init_dist == 'centered-tucker-normal':
        # The initial outputs of TuckER+ will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) + 0.5 * (init_scale ** 2)
        t = torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc
        tensor.copy_(t.float())
    elif init_dist == 'centered-cp-log-normal':
        # This initial outputs of CP^2 will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc)
        tensor.copy_(t.float())
    elif init_dist == 'centered-complex-log-normal':
        # This initial outputs of ComplEx^2 will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc)
        tensor.copy_(t.float())
    elif init_dist == 'centered-cp-log-normal-neg':
        # This initial outputs of CP^2 will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc)
        mask = torch.rand(tensor.shape[-1]) <= 0.0
        t[..., mask] = -t[..., mask]
        tensor.copy_(t.float())
    elif init_dist == 'centered-complex-log-normal-neg':
        # This initial outputs of ComplEx^2 will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc)
        mask = torch.rand(tensor.shape[-1]) <= 0.0
        t[..., mask] = -t[..., mask]
        tensor.copy_(t.float())
    elif init_dist == 'centered-rescal-log-normal':
        # This initial outputs of RESCAL^2 will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 2.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape, dtype=torch.float64) * init_scale - init_loc)
        tensor.copy_(t.float())
    else:
        raise ValueError("Unknown initializing distribution called {}".format(init_dist))


class UnnormalizedDiscreteDistribution(nn.Module):
    def __init__(self, num_categories: int, batch_size: int):
        """
        Initialize an un-normalized discrete distribution with finite number of categories.
        
        :param num_categories: The number of categories.
        :param batch_size: The batch size of the distribution.
        """
        super(UnnormalizedDiscreteDistribution, self).__init__()
        self.num_categories = num_categories
        self.batch_size = batch_size

    @property
    def num_embeddings(self):
        return self.num_categories

    @property
    def embedding_dim(self):
        return self.batch_size

    def partition_function(self):
        """
        Compute the partition function (log-space).

        :return: The partition function (log-space).
        """
        pass

    @torch.no_grad()
    def sample(self, batch_idx: torch.Tensor):
        """
        Sample from the batch distribution.

        :param batch_idx: The batch indices.
        :return: The samples.
        """
        pass


class DiscreteDistribution(UnnormalizedDiscreteDistribution):
    def __init__(self, num_categories: int, batch_size: int):
        """
        Initialize a normalized discrete distribution with finite number of categories.

        :param num_categories: The number of categories.
        :param batch_size: The batch size of the distribution.
        """
        super(DiscreteDistribution, self).__init__(num_categories, batch_size)

        # Register the constant partition function (i.e., 0 in log-space)
        self.register_buffer('z', torch.zeros(1, self.batch_size))

    def partition_function(self):
        return self.z


class Categorical(UnnormalizedDiscreteDistribution):
    def __init__(self, num_categories: int, batch_size: int, init_dist: str = 'uniform',
                 init_loc: float = 0.0, init_scale: float = 1.0):
        """
        Initialize a batch of un-normalized categorical distributions.

        :param num_categories: The number of categories.
        :param batch_size: The batch size of the distribution.
        :param init_dist: The initalizing distribution.
        :param init_loc: The initializer's hyper-parameter.
        :param init_scale: The initializer's hyper-parameter.
        """
        super(Categorical, self).__init__(num_categories, batch_size)

        # Initialize the logits parameter
        logits = torch.empty(self.num_categories, self.batch_size)
        init_params_(logits, init_dist, init_loc=init_loc, init_scale=init_scale)
        self.logits = nn.Parameter(logits, requires_grad=True)

    def partition_function(self):
        return torch.logsumexp(self.logits, dim=0, keepdim=True)

    def forward(self, x: torch.Tensor):
        return self.logits[x]

    @torch.no_grad()
    def sample(self, batch_idx: torch.Tensor):
        logits = self.logits[:, batch_idx]
        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        samples = torch.multinomial(probs.T, 1).squeeze(dim=1)
        return samples


class TwinCategorical(UnnormalizedDiscreteDistribution):
    def __init__(self, num_categories: int, batch_size: int, init_dist: str = 'uniform',
                 init_loc: float = 0.0, init_scale: float = 1.0):
        """
        Initialize a batch of un-normalized categorical distributions with two components:
        the positive part and the negative part, as in Twin-SPNs.

        :param num_categories: The number of categories.
        :param batch_size: The batch size of the distribution.
        :param init_dist: The initalizing distribution.
        :param init_loc: The initializer's hyper-parameter.
        :param init_scale: The initializer's hyper-parameter.
        """
        super(TwinCategorical, self).__init__(num_categories, batch_size)
        # Initialize the logits parameter of the positive component
        logits = torch.empty(self.num_categories, self.batch_size)
        init_params_(logits, init_dist, init_loc=init_loc, init_scale=init_scale)
        self.logits = nn.Parameter(logits, requires_grad=True)

        # Initialize the un-normalized multiplicative weights
        weight = torch.empty(self.num_categories, self.batch_size)
        init_params_(weight, 'normal', init_loc=0.0, init_scale=1e-2)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def partition_function(self):
        pos_logits = self.logits
        neg_logits = self.logits - F.softplus(-self.weight)
        pos_z = torch.logsumexp(pos_logits, dim=0, keepdim=True)
        neg_z = torch.logsumexp(neg_logits, dim=0, keepdim=True)
        return torch.stack([pos_z, neg_z], dim=2)

    def forward(self, x: torch.Tensor):
        logits = self.logits[x]
        weight = self.weight[x]
        pos_logits = logits
        neg_logits = logits - F.softplus(-weight)
        return torch.stack([pos_logits, neg_logits], dim=2)

    def sample(self, batch_idx: torch.Tensor):
        raise NotImplementedError("Batch sampling not implemented for TwinCategorical distributions")

    def twin_sample(self, pos_neg_idx: torch.Tensor, batch_idx: torch.Tensor):
        logits = self.logits[:, batch_idx]  # (n, b)
        weight = self.weight[:, batch_idx]  # (n, b)
        pos_logits = logits
        neg_logits = logits - F.softplus(-weight)
        pos_probs = torch.exp(pos_logits - torch.logsumexp(pos_logits, dim=0, keepdim=True))
        neg_probs = torch.exp(neg_logits - torch.logsumexp(neg_logits, dim=0, keepdim=True))
        probs = torch.stack([pos_probs, neg_probs], dim=0)  # (2, n, b)
        probs = probs[pos_neg_idx, :, torch.arange(len(batch_idx), device=batch_idx.device)]  # (b, n)
        samples = torch.multinomial(probs, 1).squeeze(dim=1)
        return samples
