from typing import Optional

import torch
from torch import nn

LOSSES_NAMES = ['LCWA+ce', 'NLL']


def average_nll(x: torch.Tensor, y: Optional[torch.Tensor] = None):
    return -torch.mean(x)


def setup_loss(loss: str):
    if loss == 'LCWA+ce':
        return nn.CrossEntropyLoss(reduction='mean')
    if loss == 'NLL':
        return average_nll
    raise ValueError("Unknown loss setting called {}".format(loss))
