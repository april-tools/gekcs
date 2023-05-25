import random

import torch
import numpy as np


def set_seed(seed: int, is_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def average_metrics_entity(mrrs: dict, hits: dict):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.0
    h = (hits['lhs'] + hits['rhs']) / 2.0
    return {'mrr': m, 'hits@[1,3,10]': h.tolist()}


def safelog(x: torch.Tensor, eps: float = 1e-15):
    return torch.log(torch.clamp(x, eps))
