from __future__ import annotations
from typing import List
import numpy as np
import torch


class Game:
    @staticmethod
    def make_move(board, action):
        raise NotImplementedError

    @staticmethod
    def get_init_board():
        raise NotImplementedError

    @staticmethod
    def get_legal_actions(state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def is_terminal(board) -> (bool, int):
        raise NotImplementedError


class MCTSNode:
    def __init__(
        self,
        state: torch.Tensor,
        prior: float = 0,
        action: torch.Tensor = torch.Tensor([]),
    ):

        self._action = action
        self._state = state
        self._children: List[MCTSNode] = []
        self._value = None
        self._policy = None
        self.n_visit = 0
        self.total_value = 0
        self.is_terminal = False
        self.prior = prior
        self.parent = None

    def add_child(self, child: MCTSNode):
        self._children.append(child)
        child.parent = self

    def calc_policy_value(self, network: torch.nn.Module):
        # TODO: SOFTMAX ONLY FOR LEGAL ACTIONS, return logits?
        self._policy, self._value = network(self._state)

    def expand(self, legal_actions):
        for index in legal_actions.nonzero():
            action = torch.zeros(legal_actions.size())
            action[index] = 1
            prior = self._policy[index].item()
            child = MCTSNode(
                state=Game.make_move(self._state, action), action=action, prior=prior
            )
            self.add_child(child)

    @property
    def is_leaf(self):
        return self.n_visit == 0

    @property
    def q_value(self):
        return self.total_value / self.n_visit

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def children(self):
        return self._children

    @property
    def is_root(self):
        return self.parent is None

    def u_value(self, sum_n: int, c_puct: float = 1.0):
        return c_puct * self.prior * np.sqrt(sum_n) / (1 + self.n_visit)

    def confidence_bound(self, sum_n: int, c_puct: float = 1.0):
        return self.u_value(sum_n, c_puct) + self.q_value


def mcts(
    start_node: MCTSNode,
    game: Game,
    net: torch.nn.Module,
    n_iter: int = 1600,
    c_puct: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    def backpropagate(leaf_node: MCTSNode):
        _, v_leaf = leaf_node.calc_policy_value(
            network=net
        )  # Think about batching/efficiency
        leaf_node.is_terminal, _ = Game.is_terminal(leaf_node.state)
        leaf_node.n_visit += 1
        leaf_node.total_value += v_leaf
        if not leaf_node.is_terminal:
            # Expand node
            legal_actions = game.get_legal_actions(leaf_node.state)
            leaf_node.expand(legal_actions)
        current = leaf_node.parent
        while not current.is_root:
            # Don't need to add to root as parent/action combo is stored as child node instead here
            current.n_visit += 1
            current.value += v_leaf
            current = current.parent

    def forward(root_node: MCTSNode):
        current = root_node
        while not current.is_leaf and not current.is_terminal:
            # Select child with highest cb
            sum_n = np.sum([child.n_visit for child in current.children])
            current = current.children[
                np.argmax(
                    [
                        child.confidence_bound(sum_n=sum_n, c_puct=c_puct)
                        for child in current.children
                    ]
                )
            ]
        backpropagate(current)

    for _ in range(n_iter):
        forward(start_node)
    policy_count = torch.LongTensor([child.n_visit for child in start_node.children])
    sampled_node = start_node.children[
        torch.multinomial(policy_count.power(1 / temperature), 1).item()
    ]
    # TODO Maybe just iterate through children once, more efficient?
    # Currently doing it twice due to output difference, just do forloop instead?
    return (
        sum([child.action * child.n_visit for child in start_node.children]),
        sampled_node,
    )


def self_play(
    game: Game, net: torch.nn.Module, n_mcts_iter: int = 1600,
) -> List[tuple]:
    pos = game.get_init_board()
    temperature = 0.0001
    data = []
    is_terminal, res = game.is_terminal(pos)
    node = MCTSNode(state=pos)
    while not is_terminal:
        policy, node = mcts(
            start_node=node,
            game=game,
            net=net,
            n_iter=n_mcts_iter,
            temperature=temperature,
        )
        data.append((pos, policy))
        pos = game.make_move(pos, node.action)
        is_terminal, res = game.is_terminal(pos)
        # TODO: Currently does not add last state (when game ends, no moves available) to datapoints, should it? Also look into stopping thresholds and resigning
    data = [row + (res,) for row in data]

    return data


def play_games(
    game: Game,
    net: torch.nn.Module,
    n_mcts_iter: int = 1600,
    n_games_thread=2500,
    threads=4,
):
    # Arg games/thread instead?
    def play_games_thread():
        data = []
        for i in range(n_games_thread):
            
