from alphazero.mcts import mcts, MCTSNode, self_play
from alphazero import TicTacToe, PhilipNet, ResidualBlock, TicTacToeConverter
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
        ttt_instance = TicTacToe(3, 3)
        dummy_net = PhilipNet(ttt_instance, ResidualBlock, 3, 3,)
        mcts(
            start_node=MCTSNode(
                state=TicTacToe.get_initial_board(3, 3), player=0,
            ),
            rollout=False,
            net=dummy_net,
            game=TicTacToe,
            board_converter=TicTacToeConverter,
            n_iter=1000,
            c_puct=5.0,
        )

    def test_self_play_runs(self):
        # Just check that it runs for now

        ttt_instance = TicTacToe(3, 3)
        dummy_net = PhilipNet(ttt_instance, ResidualBlock, 3, 3,)

        self_play(
            game=TicTacToe,
            board_converter=TicTacToeConverter,
            net=dummy_net,
            n_mcts_iter=10,
        )
