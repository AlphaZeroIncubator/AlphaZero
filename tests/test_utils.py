#! /usr/bin/python3

from alphazero import AlphaGoZeroLoss
import torch


class Test_Loss:
    def test_alphazero_input(self):
        policy_true = torch.Tensor([0.1, 0.1, 0.8])
        q_true = torch.Tensor([[1000, 0]])

        policy_pred = torch.Tensor([0.1, 0.1, 0.8])
        q_pred = torch.LongTensor([0])

        truth = (policy_true, q_true)
        pred = (policy_pred, q_pred)

        assert torch.allclose(AlphaGoZeroLoss(truth, pred), torch.Tensor([0]))

        policy_pred = torch.Tensor([1.0, 0, 0])
        q_pred = torch.LongTensor([1])

        pred = (policy_pred, q_pred)

        assert torch.allclose(
            AlphaGoZeroLoss(truth, pred), torch.Tensor([1000.4867])
        )
