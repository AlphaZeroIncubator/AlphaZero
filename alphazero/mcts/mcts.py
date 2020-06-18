"""Functions and classes for MCTS implementation."""
from __future__ import annotations

from typing import List, TypeVar

from alphazero.utils import sample_tensor_indices

import numpy as np

import torch


GameClassType = TypeVar("T", bound="Game")


class MCTSNode:
    """Node Class for MCTS tree search."""

    def __init__(
        self,
        state: torch.Tensor,
        player: int = 0,
        root_player: int = 0,
        prior: float = 0,
        action: torch.Tensor = torch.Tensor([]),
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
        self.root_player = root_player

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
        # Should probably update below logic in game class, cu
        return (
            game.result(state)
            if self.root_player == 0
            else -game.result(state)
        )

    def calc_policy_value(
        self,
        rollout: bool,
        game: GameClassType,
        network: torch.nn.Module = None,
    ) -> (torch.Tensor, float):
        """Calculate the policy and value for the node.

        Calculate the policy and value for the node. If rollout = True,
        performs a rollout, else uses network to get policy and value.
        Args:
        rollout (bool): Whether to perform a rollout or not
        game (GameClassType): Game that is played
        network (torch.nn.Module, optional): Must be set if rollout is false

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
        """Expand node.

        Args:
            game (GameClassType): Game that is played
        """
        n_players = game.n_players()
        legal_actions = game.get_legal_moves(self._state)
        for index in legal_actions.nonzero():
            action = torch.zeros(legal_actions.size())
            action[tuple(index)] = 1
            prior = self._policy[tuple(index)].item()
            child = MCTSNode(
                state=game.board_after_move(
                    self._state, self.player, tuple(index)
                ),
                player=(self.player + 1) % n_players,
                action=action,
                prior=prior,
                root_player=self.root_player,
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

    def u_value(self, sum_n: int, c_puct: float = 1.0):
        """Calculate U for the puct calculation.

        Args:
            sum_n (int): Sum of all visits to children from parent
            c_puct (float, optional): Puct parameter. Defaults to 1.0.

        Returns:
            float: U value
        """
        return c_puct * self.prior * np.sqrt(sum_n) / (1 + self.n_visit)

    def puct(self, sum_n: int, c_puct: float = 1.0):
        """Calculate puct for the mcts.

        Args:
            sum_n (int): Sum of all visits to children from parent
            c_puct (float, optional): Puct parameter. Defaults to 1.0.

        Returns:
            float: puct value
        """
        return self.u_value(sum_n, c_puct) + self.q_value

    # Make root function here (delete parent tree, set root_player to player)


def mcts(
    start_node: MCTSNode,
    game: GameClassType,
    rollout=False,
    net: torch.nn.Module = None,
    n_iter: int = 1600,
    c_puct: float = 1.0,
    temperature: float = 1.0,
    # eval_func: str = "PUCT",
) -> (torch.Tensor, MCTSNode):
    # TODO USE EVAL VARIABLE
    """Run mcts search from start node.

    Args:
        start_node (MCTSNode): Node to search from
        game (GameClassType): Game that is played
        rollout (bool, optional): Whether the search uses rollout instead of
        a network. Defaults to False.
        net (torch.nn.Module, optional): Needs to be input if rollout = false
        n_iter (int, optional): How many iterations to run Defaults to 1600.
        c_puct (float, optional): c_puct value for node evaluation.
        Defaults to 1.0.
        temperature (float, optional): Temperature for policy calculation.
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
        """Run a forward pass through the tree.

        Run a forward pass through the tree and then backpropagates
        back through the tree.
        Args:
        root_node (MCTSNode): Root node to start from
        """
        current = root_node
        while not current.is_leaf and not current.is_terminal:
            sum_n = np.sum([child.n_visit for child in current.children])
            current = current.children[
                np.argmax(
                    [
                        child.puct(sum_n=sum_n, c_puct=c_puct)
                        for child in current.children
                    ]
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
    return (
        sum([child.action * child.n_visit for child in start_node.children]),
        sampled_node,
    )
