import pytest
from alphazero.mcts import mcts, MCTSNode
from alphazero import TicTacToe
import torch


class TestMCTS:
    def test_tictactoe_first_step(self):
        policy, _ = mcts(
            start_node=MCTSNode(state=TicTacToe.get_initial_board(3, 3)),
            rollout=True,
            game=TicTacToe,
            n_iter=1000,
            c_puct=5.0,
        )
        assert torch.argmax(policy).item() == 4  # Middle
