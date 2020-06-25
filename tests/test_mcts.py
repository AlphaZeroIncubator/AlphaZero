from alphazero.mcts import mcts, MCTSNode, self_play
from alphazero import TicTacToe
import torch


class TestMCTS:
    def test_tictactoe_first_step(self):
        policy, _ = mcts(
            start_node=MCTSNode(
                state=TicTacToe.get_initial_board(3, 3), player=0,
            ),
            rollout=True,
            game=TicTacToe,
            n_iter=10000,
            c_puct=5.0,
            dirichlet_eps=0,
            dirichlet_conc=1,
        )
        assert torch.argmax(policy).item() == 4  # Middle

    def test_mcts_net(self):
        # Just check that it runs for now
        def dummy_net(inp):
            return torch.ones(3, 3) / 9, torch.tensor(0)

        policy, _ = mcts(
            start_node=MCTSNode(
                state=TicTacToe.get_initial_board(3, 3), player=0,
            ),
            rollout=False,
            net=dummy_net,
            game=TicTacToe,
            n_iter=1000,
            c_puct=5.0,
        )

    def test_self_play(self):
        # Just check that it runs for now
        def dummy_net(inp):
            return torch.ones(3, 3) / 9, torch.tensor(0)

        data = self_play(game=TicTacToe, net=dummy_net)
