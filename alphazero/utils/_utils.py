"""Util functions for alphazero."""
from typing import List

import numpy as np

import torch

import torch.nn as nn


def sample_tensor_indices(
    tensor: torch.Tensor, n_samples: int = 1
) -> List[tuple]:
    """Do multinomial sampling on tensor, returning indices.

    Args:
    tensor (torch.Tensor): Tensor to be sampled
    n_samples (int, optional): How many samples (no replacement).
    Defaults to 1.

    Returns:
    List[tuple]: List of sampled indices
    """
    ind_1d = tensor.float().flatten().multinomial(n_samples)
    samples = np.unravel_index(ind_1d, tensor.size())
    samples = np.array(samples)
    return [tuple(r) for r in samples.T]


def AlphaGoZeroLoss(
    ground_truth: (torch.Tensor, torch.Tensor),
    output: (torch.Tensor, torch.Tensor),
    ratio: float = 1.0,
):
    """
    Calculates AlphaGoZero style loss.
    L2 regularization is included in the optimizer and would be very slow to
    compute here, since it would require unpacking the parameters of the NN
    at every step. So we're using it there instead.
    """

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    policy_true = ground_truth[0]
    q_true = ground_truth[1]

    policy = output[0]
    q_value = output[1]

    return mse(policy_true, policy) + ratio * ce(q_true, q_value)

