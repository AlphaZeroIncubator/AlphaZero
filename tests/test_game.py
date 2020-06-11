#! /usr/bin/python3

import pytest
from alphazero import *


class Test_TicTacToe:
    @classmethod
    def setup_class(cls):
        cls.ttt = TicTacToe()

    def test_example(self):
        assert self.ttt.board_state.shape[0] > 1
