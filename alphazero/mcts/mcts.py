from __future__ import annotations
from typing import List, TypeVar
import numpy as np
import torch
from alphazero import TicTacToe
from alphazero.utils import sample_tensor_indices

GameClassType = TypeVar("T", bound="Game")


class BoardConverter:
    @staticmethod
    def board_to_tensor(board):
        raise NotImplementedError

    @staticmethod
    def action_to_tensor(action):
        # Not sure if needed
        raise NotImplementedError

    @staticmethod
    def tensor_to_action(action_tensor):
        raise NotImplementedError


class TicTacToeConverter(BoardConverter):
    @staticmethod
    def board_to_tensor(board):
        player_1 = board == 0
        player_2 = board == 1
        player = 1 if player_1.sum() > player_2.sum() else 0
        player_layer = torch.full(board.size(), player)
        return torch.stack((player_1, player_2, player))

    ## TODO: Flesh this out and move


class MCTSNode:
    """
    Node Class for MCTS tree search
    """

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
        """
        Adds a child to node instance

        Args:
            child (MCTSNode): Node to be added as child
        """
        self._children.append(child)
        child.parent = self

    def rollout(self, game: GameClassType) -> int:
        """
        Performs a rollout from self, selecting random actions from a uniform prior until game ends
        Args:
            game (GameClassType): Game that is currently played

        Returns:
            result (int): Final result of rollout
        """
        state = self._state
        while not game.is_game_over(state):
            legal_actions = game.get_legal_moves(state)
            action = sample_tensor_indices(legal_actions, 1)[0]
            player = 1 if (state == 0).sum() > (state == 1).sum() else 0
            state = game.board_after_move(state, player, action)
        result = game.result(state)
        return result

    def calc_policy_value(
        self, rollout: bool, game: GameClassType, network: torch.nn.Module = None
    ) -> (torch.Tensor, float):
        """
        Calculates the policy and value for the node. If rollout performs a rollout, else uses network to get policy and value
        Args:
            rollout (bool): Whether to perform a rollout or not
            game (GameClassType): Game that is played
            network (torch.nn.Module, optional): Optional, must be set if rollout is false

        Returns:
            policy (torch.Tensor): The calculated policy
            value (float): The calculated value
        """
        if not rollout:
            # TODO: Normalize probabilities after filter legal actions
            self._policy, self._value = network(self._state)
        else:
            self._value = self.rollout(game)
            policy = game.get_legal_moves(self._state)
            self._policy = policy.float() / policy.sum()
        return self._policy, self._value

    def expand(self, game: GameClassType):
        """
        Expands node
        Args:
            game (GameClassType): Game that is played
        """
        legal_actions = game.get_legal_moves(self._state)
        for index in legal_actions.nonzero():
            action = torch.zeros(legal_actions.size())
            action[tuple(index)] = 1
            prior = self._policy[tuple(index)].item()
            player = 1 if (self._state == 0).sum() > (self._state == 1).sum() else 0
            child = MCTSNode(
                state=game.board_after_move(self._state, player, tuple(index)),
                action=action,
                prior=prior,
            )
            self.add_child(child)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def q_value(self):
        if self.n_visit > 0:
            return self.total_value / self.n_visit
        else:
            return 0

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
    game: GameClassType,
    rollout=False,
    net: torch.nn.Module = None,
    n_iter: int = 1600,
    c_puct: float = 1.0,
    temperature: float = 1.0,
    eval: str = "PUCT",
) -> (torch.Tensor, MCTSNode):
    # TODO USE EVAL VARIABLE
    """
    Runs mcts search from start node
    Args:
        start_node (MCTSNode): Node to search from
        game (GameClassType): Game that is played
        rollout (bool, optional): Whether the search uses rollout instead of a network. Defaults to False.
        net (torch.nn.Module, optional): Needs to be input if rollout = false
        n_iter (int, optional): How many iterations to run Defaults to 1600.
        c_puct (float, optional): c_puct value for node evaluation. Defaults to 1.0.
        temperature (float, optional): Temperature for policy calculation. Defaults to 1.0.
        eval (str, optional): Evaluation type. Defaults to "PUCT".
    Returns:
        policy (torch.Tensor)
        sampled_node (MCTSNode)
    """

    def backpropagate(leaf_node: MCTSNode):
        """
        Backpropagates up the tree after reaching leaf node
        Args:
            leaf_node (MCTSNode): Leaf node to backpropagate from
        """
        if not rollout:
            _, v_leaf = leaf_node.calc_policy_value(
                rollout=rollout, network=net
            )  # Think about batching/efficiency
        else:
            _, v_leaf = leaf_node.calc_policy_value(rollout=rollout, game=game)

        leaf_node.is_terminal = game.is_game_over(leaf_node.state)
        if not leaf_node.is_terminal:
            leaf_node.expand(game)
        current = leaf_node
        while not current.is_root:
            current.n_visit += 1
            current.total_value += v_leaf
            current = current.parent

    def forward(root_node: MCTSNode):
        """
        Runs a forward pass through the tree and then backpropagates back through the tree
        Args:
            root_node (MCTSNode): Root node to start from
        """
        current = root_node
        while not current.is_leaf and not current.is_terminal:
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
        torch.multinomial(torch.pow(policy_count, (1 / temperature)).float(), 1).item()
    ]
    # TODO Maybe just iterate through children once, more efficient?
    # Currently doing it twice due to output difference, just do forloop instead?
    return (
        sum([child.action * child.n_visit for child in start_node.children]),
        sampled_node,
    )


def self_play(
    game: GameClassType, net: torch.nn.Module, n_mcts_iter: int = 1600,
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


# def play_games(
#    game: Game,
#    net: torch.nn.Module,
#    n_mcts_iter: int = 1600,
#    n_games_thread=2500,
#    threads=4,
# ):
#    # Arg games/thread instead?
#    def play_games_thread():
#        data = []
#        for i in range(n_games_thread):

