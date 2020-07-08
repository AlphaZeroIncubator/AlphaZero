from alphazero.model import PhilipNet
from alphazero.mcts import self_play
from alphazero import Connect4
from alphazero import TicTacToe
from alphazero import LeoConverter

n_mcts_iter = 100
temperature = 0.5
dirichlet_eps = 0.25
dirichlet_conc = 1.0

game = TicTacToe()

model = PhilipNet(in_channels = 2, game = game, num_blocks = 10, enc_channels = 128)
# print(model(LeoConverter.board_to_tensor(game.board_state,1)))

my_game = self_play(
        game,
        model,
        board_converter=LeoConverter,
        n_mcts_iter=n_mcts_iter,
        temperature=temperature,
        dirichlet_eps=dirichlet_eps,
        dirichlet_conc=dirichlet_conc,
        )
