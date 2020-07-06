import numpy as np
import torch
import torch.nn as nn
from alphazero import load_board
from alphazero.mcts import self_play
from copy import deepcopy


def train(
    game,
    model,
    optimizer,
    board_converter,
    n_games=1000,
    subset_size=1000,
    n_mcts_iter=1600,
    temperature=1e-4,
    dirichlet_eps=0.25,
    dirichlet_conc=1.0,
    device=torch.device("cpu"),
    n_test_games=9,
):

    data = []
    # generate data through self-play with MCTS
    for i in range(n_games):
        data.append(
            self_play(
                game,
                model,
                board_converter=board_converter,
                n_mcts_iter=n_mcts_iter,
                temperature=temperature,
                dirichlet_eps=dirichlet_eps,
                dirichlet_conc=dirichlet_conc,
            )
        )

    # train model on random positions from generated data
    subset = np.random.choice(data, size=subset_size)

    old_model = deepcopy(model)

    model.to(device)
    model.train()
    for idx, (pos, policy, res) in enumerate(subset):
        # send to device
        pos, policy, res = (
            pos.to(device),
            policy.to(device),
            res.to(device),
        )

        optimizer.zero_grad()

        output = model(pos)

        loss = AlphaGoZeroLoss((policy, res), output)
        loss.backward()

        optimizer.step()

        if idx % 100 == 0:
            print(
                "  [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    idx, subset_size, 100.0 * idx / subset_size, loss.item()
                )
            )

    # evaluate if model is better than old model (through self-play)

    model = test(game, model, old_model, n_games=n_test_games, device=device)


def test(game_type, model_1, model_2, n_games=9, device=torch.device("cpu")):
    tally = 0

    for game in range(n_games):
        tally += play_game(game_type, model_1, model_2, device)

    print(f"New model against old model tally: {tally}")

    if tally > 0:
        return model_1
    # implicit else: if the model_1 ties with the model_2, we keep model_2
    return model_2


def play_game(game_type, model_1, model_2, device=torch.device("cpu")):

    game = game_type()
    model_1.to(device)
    model_2.to(device)
    current_player = model_1

    while not game_type.is_game_over(game.board_state):
        # play the moves individually
        pos = game.board_state.to(device)
        moves, q = current_player(pos)
        game.make_move(np.argmax(moves))

        # i'm not confident this works since it relies on "is"
        current_player = model_2 if current_player is model_1 else model_1

    # return the winner of the game
    return game_type.result(game.board_state)
