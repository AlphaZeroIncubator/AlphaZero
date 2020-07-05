"""Functions and classes for MCTS implementation."""
from __future__ import annotations

from typing import List, TypeVar, Tuple

from alphazero.utils import sample_tensor_indices

from alphazero import BoardConverter, Game  # noqa

import numpy as np

import torch

from collections import namedtuple

GameClassType = TypeVar("GameClassType", bound="Game")

BoardConverterType = TypeVar("BoardConverterType", bound="BoardConverter")


class MCTSNode:
    """Node Class for MCTS tree search."""

    def __init__(
        self,
        state: torch.Tensor,
        player: int = 0,
        prior: float = 0,
        action: torch.Tensor = None,
    ):
        """Create an MCTSNode.

        Args:
            state (torch.Tensor): Board state for node
            prior (float, optional): Prior for selecting this action/node
            from parent. Defaults to 0.
            action (torch.Tensor, optional): Action from parent to this node.
            Defaults to torch.Tensor([]).
        """
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
        self.player = player

    def add_child(self, child: MCTSNode):
        """Add a child to node instance.

        Args:
            child (MCTSNode): Node to be added as child
        """
        self._children.append(child)
        child.parent = self

    def rollout(self, game: GameClassType) -> int:
        """Perform a rollout from self.

        Perform a rollout from self, selecting random actions from a uniform
        prior until game ends.
        Args:
        game (GameClassType): Game that is currently played

        Returns:
            result (int): Final result of rollout
        """
        state = self._state
        player = self.player
        n_players = game.n_players()
        while not game.is_game_over(state):
            legal_actions = game.get_legal_moves(state)
            action = sample_tensor_indices(legal_actions, 1)[0]
            state = game.board_after_move(state, player, action)
            player = (player + 1) % n_players

        return game.result(state, player), player

    def calc_policy_value(
        self,
        rollout: bool,
        game: GameClassType,
        board_converter: BoardConverterType = None,
        network: torch.nn.Module = None,
    ) -> (torch.Tensor, float):
        """Calculate the policy and value for the node.

        Calculate the policy and value for the node. If rollout = True,
        performs a rollout, else uses network to get policy and value.
        Args:
        rollout (bool): Whether to perform a rollout or not
        game (GameClassType): Game that is played
        board_converter (BoardConverterType): Class for converting board state
        to input tensor
        network (torch.nn.Module, optional): Must be set if rollout is false

        Returns:
            policy (torch.Tensor): The calculated policy
            value (float): The calculated value
        """
        if not rollout:
            policy, value = network(
                board_converter.board_to_tensor(self._state, self.player)
            )
            policy = policy * game.get_legal_moves(self._state).float()
            policy = policy / policy.sum()
            self._policy, self._value = policy, value.item()
        else:
            value, val_player = self.rollout(game)
            node_value = value if val_player == self.player else -value
            self._value = node_value
            policy = game.get_legal_moves(self._state)
            self._policy = policy.float() / policy.sum()
        return self._policy, self._value

    def expand(
        self,
        game: GameClassType,
        dirichlet_eps: float = 0.25,
        dirichlet_conc: float = 0.03,
    ):
        """Expand node.

        Args:
            game (GameClassType): Game that is played
        """
        n_players = game.n_players()
        legal_actions = game.get_legal_moves(self._state)
        if self.is_root:
            n = legal_actions.sum().item()
            dir_noise = np.random.dirichlet(dirichlet_conc * np.ones(n), 1)[0]
        for i, index in enumerate(legal_actions.nonzero()):
            action = torch.zeros(legal_actions.size())
            action[tuple(index)] = 1
            prior = self._policy[tuple(index)].item()
            if self.is_root:
                prior = (
                    prior * (1 - dirichlet_eps) + dirichlet_eps * dir_noise[i]
                )
            child = MCTSNode(
                state=game.board_after_move(
                    self._state, self.player, tuple(index)
                ),
                player=(self.player + 1) % n_players,
                action=action,
                prior=prior,
            )
            self.add_child(child)

    @property
    def is_leaf(self):
        """Property for whether a node is a leaf or not.

        Returns:
            bool: Whether the node is a leaf or not
        """
        return len(self.children) == 0

    @property
    def q_value(self):
        """Calculate the Q values for puct.

        Returns:
            float: The q value
        """
        if self.n_visit > 0:
            return self.total_value / self.n_visit
        else:
            return 0

    @property
    def state(self) -> torch.Tensor:
        """Property for the node state.

        Returns:
            torch.Tensor: The node state
        """
        return self._state

    @property
    def action(self) -> torch.Tensor:
        """Property for the nodes action, i.e. edge to parent.

        Returns:
            torch.Tensor: The node action
        """
        return self._action

    @property
    def children(self) -> List[MCTSNode]:
        """Property for the nodes children.

        Returns:
            List[MCTSNode]: The nodes children as a list
        """
        return self._children

    @property
    def is_root(self) -> bool:
        """Whether the node is a root node.

        Returns:
            bool: Whether the node is a root node
        """
        return self.parent is None

    def u_value(self, c_puct: float = 1.0):
        """Calculate U for the puct calculation.

        Args:
            sum_n (int): Sum of all visits to children from parent
            c_puct (float, optional): Puct parameter. Defaults to 1.0.

        Returns:
            float: U value
        """
        return (
            c_puct
            * self.prior
            * np.sqrt(self.parent.n_visit)
            / (1 + self.n_visit)
        )

    def puct(self, c_puct: float = 1.0):
        """Calculate puct for the mcts.

        Args:
            sum_n (int): Sum of all visits to children from parent
            c_puct (float, optional): Puct parameter. Defaults to 1.0.

        Returns:
            float: puct value
        """
        return self.u_value(c_puct) - self.q_value

    # Make root function here (delete parent tree)


def mcts(
    start_node: MCTSNode,
    game: GameClassType,
    board_converter: BoardConverterType = None,
    rollout=False,
    net: torch.nn.Module = None,
    n_iter: int = 1600,
    c_puct: float = 1.0,
    temperature: float = 1.0,
    dirichlet_eps: float = 0.25,
    dirichlet_conc: float = 1.0
    # eval_func: str = "PUCT",
) -> Tuple[torch.Tensor, MCTSNode]:
    # TODO USE EVAL VARIABLE
    """Run mcts search from start node.

    Args:
        start_node (MCTSNode): Node to search from
        game (GameClassType): Game that is played
        board_converter (BoardConverterType): Class to convert board state to
        neural net input
        rollout (bool, optional): Whether the search uses rollout instead of
        a network. Defaults to False.
        net (torch.nn.Module, optional): Needs to be input if rollout = false
        n_iter (int, optional): How many iterations to run Defaults to 1600.
        c_puct (float, optional): c_puct value for node evaluation.
        Defaults to 1.0.
        temperature (float, optional): Temperature for policy calculation.
        Defaults to 1.0.
        dirichlet_eps (float, optional): Dirichlet noise epsilon.
        Defaults to 0.25.
        dirichlet_conc (float, optional): Dirichlet noise concentration.
        Defaults to 1.0.
        eval (str, optional): Evaluation type. Defaults to "PUCT".

    Returns:
        policy (torch.Tensor)
        sampled_node (MCTSNode)
    """
    if not rollout and net is None:
        raise ValueError("Rollout is set to false but no network was provided")

    def backpropagate(leaf_node: MCTSNode):
        """
        Backpropagates up the tree after reaching leaf node.

        Args:
        leaf_node (MCTSNode): Leaf node to backpropagate from
        """
        if not rollout:
            _, v_leaf = leaf_node.calc_policy_value(
                rollout=rollout,
                network=net,
                game=game,
                board_converter=board_converter,
            )  # Think about batching/efficiency
        else:
            _, v_leaf = leaf_node.calc_policy_value(rollout=rollout, game=game)
        leaf_player = leaf_node.player
        leaf_node.is_terminal = game.is_game_over(leaf_node.state)
        if not leaf_node.is_terminal:
            leaf_node.expand(
                game,
                dirichlet_eps=dirichlet_eps,
                dirichlet_conc=dirichlet_conc,
            )
        current = leaf_node
        while current is not None:
            current.n_visit += 1
            current.total_value += (
                v_leaf if current.player == leaf_player else -v_leaf
            )
            current = current.parent

    def forward(root_node: MCTSNode):
        """Run a forward pass through the tree.

        Run a forward pass through the tree and then backpropagates
        back through the tree.
        Args:
        root_node (MCTSNode): Root node to start from
        """
        current = root_node
        while not current.is_leaf and not current.is_terminal:
            current = current.children[
                np.argmax(
                    [child.puct(c_puct=c_puct) for child in current.children]
                )
            ]
        backpropagate(current)

    for _ in range(n_iter):
        forward(start_node)

    policy_count = torch.LongTensor(
        [child.n_visit for child in start_node.children]
    )

    sampled_node = start_node.children[
        torch.multinomial(
            torch.pow(policy_count, (1 / temperature)).float(), 1
        ).item()
    ]
    # TODO Maybe just iterate through children once, more efficient?
    # Currently doing it twice due to output difference,
    # just do forloop instead?
    # This can def be optimized better
    return (
        torch.pow(
            sum(
                [child.action * child.n_visit for child in start_node.children]
            ),
            (1 / temperature),
        ),
        sampled_node,
    )


def self_play(
    game: GameClassType,
    net: torch.nn.Module,
    board_converter: BoardConverterType,
    n_mcts_iter: int = 1600,
    temperature: float = 1.0,
    dirichlet_eps: float = 0.25,
    dirichlet_conc: float = 1.0,
) -> List[tuple]:
    """Plays games with itself.

    Args:
        game (GameClassType): The game class for the model
        net (torch.nn.Module): The model network
        board_converter (BoardConverterType): Converts board state
        to input tensor.
        n_mcts_iter (int, optional): Number of mcts iterations.
        Defaults to 1600.
        temperature (float, optional): Temperature for probability calculation.
        Defaults to 1.0.
        dirichlet_eps (float, optional): Dirichlet noise epsilon.
        Defaults to 0.25.
        dirichlet_conc (float, optional): Dirichlet noise concentration.
        Defaults to 1.0.

    Returns:
        List[tuple]: List of data point tuples (pos, policy, result)
    """

    SelfPlayPoint = namedtuple("SelfPlayPoint", "pos policy player")

    pos = game.get_initial_board()
    data = []
    is_terminal = game.is_game_over(pos)
    player = 0
    node = MCTSNode(state=pos, player=player)
    n_players = game.n_players()
    while not is_terminal:
        policy, node = mcts(
            start_node=node,
            game=game,
            board_converter=board_converter,
            net=net,
            n_iter=n_mcts_iter,
            temperature=temperature,
            dirichlet_eps=dirichlet_eps,
            dirichlet_conc=dirichlet_conc,
        )
        node.parent = None  # Make new node root, keep sub-tree information
        data.append(SelfPlayPoint(pos, policy, player))
        pos = node.state
        player = (player + 1) % n_players
        is_terminal = node.is_terminal
        # TODO: Look into stopping thresholds and resigning
    res = game.result(pos, player)

    # Returns (pos, policy, res) tuples in list.
    # res is flipped if game result was computed
    # from opponents point of view
    return [
        (row.pos, row.policy) + (res if row.player == player else -res,)
        for row in data
    ]
