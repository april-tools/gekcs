import torch
from kbc.datasets import TypedDataset


if __name__ == '__main__':
    # Validate the consistency of the training KG in the 'ogbl-biokg' dataset
    data = TypedDataset("ogbl-biokg", torch.device("cuda"), reciprocal=False)
    assert data.validate_train_split()
