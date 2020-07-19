#! /usr/bin/python3

import torch
from alphazero.utils import sample_tensor_indices, AlphaGoZeroLoss


class TestUtils:
    def test_sampling(self):
        tensor = torch.zeros((3, 4, 5))
        tensor[1, 2, 3] = 1
        res = sample_tensor_indices(tensor, 1)
        assert res[0] == (1, 2, 3)


class Test_Loss:
    def test_alphazero_input(self):
        policy_true = torch.Tensor([0.1, 0.1, 0.8])
        q_true = torch.Tensor([[0]])

        policy_pred = torch.Tensor([0.1, 0.1, 0.8])
        q_pred = torch.LongTensor([0])

        truth = (policy_true, q_true)
        pred = (policy_pred, q_pred)

        assert torch.allclose(
            AlphaGoZeroLoss(truth, pred), torch.Tensor([0.689727])
        )

        policy_pred = torch.Tensor([1.0, 0, 0])
        q_pred = torch.LongTensor([0])

        pred = (policy_pred, q_pred)

        assert torch.allclose(
            AlphaGoZeroLoss(truth, pred), torch.Tensor([1.55144])
        )

        policy_pred = torch.Tensor([0.1, 0.1, 0.8])
        q_pred = torch.LongTensor([2])

        pred = (policy_pred, q_pred)

        assert torch.allclose(
            AlphaGoZeroLoss(truth, pred), torch.Tensor([4.689727])
        )
