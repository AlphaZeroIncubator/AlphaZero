#! /usr/bin/python3

import torch
import torch.nn as nn


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
