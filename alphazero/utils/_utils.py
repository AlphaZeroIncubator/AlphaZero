"""Util functions for alphazero."""
from typing import List

import numpy as np

import torch


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
