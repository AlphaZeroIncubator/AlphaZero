import pytest
from alphazero.utils import *
import torch


class TestUtils:
    def test_sampling(self):
        tensor = torch.zeros((3, 4, 5))
        tensor[1, 2, 3] = 1
        res = sample_tensor_indices(tensor, 1)
        assert res[0] == (1, 2, 3)
