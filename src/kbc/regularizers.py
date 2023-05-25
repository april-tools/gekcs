from typing import List, Optional, Union

import torch

REGULARIZERS_NAMES = ['None', 'F2', 'N3']


class F2:
    def __init__(self, lmbda: float):
        if lmbda <= 0.0:
            raise ValueError("The lambda regularization factor for F2 must be positive")
        self.lmbda = lmbda

    def penalty(self, factors: List[torch.Tensor]) -> torch.Tensor:
        """
        Frobenius Norm (F2) of factors.
        """
        norm = 0.0
        for f in factors:
            norm += torch.sum(torch.square(f))
        return self.lmbda * norm / factors[0].shape[0]


class N3:
    def __init__(self, lmbda: float):
        if lmbda <= 0.0:
            raise ValueError("The lambda regularization factor for N3 must be positive")
        self.lmbda = lmbda

    def penalty(self, factors: List[torch.Tensor]) -> torch.Tensor:
        """
        Weighted nuclear 3-norm (N3) of factors.
        """
        norm = 0.0
        for f in factors:
            norm += torch.sum(torch.abs(f) ** 3)
        return self.lmbda * norm / factors[0].shape[0]


def setup_regularizer(regularizer: str, lmbda: float) -> Optional[Union[F2, N3]]:
    if regularizer == 'F2':
        return F2(lmbda)
    if regularizer == 'N3':
        return N3(lmbda)
    if regularizer == 'None':
        return None
    raise ValueError("Unknown regularizer called {}".format(regularizer))
