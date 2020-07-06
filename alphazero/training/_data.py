import os

import torch

from alphazero import load_board


class SelfPlayDataset(torch.utils.data.Dataset):
    # untested
    def __init__(self, game, root_dir, transform):
        self.game = game
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return sum(
            [
                len(os.listdir(directory))
                for directory in os.listdir(self.root_dir)
            ]
        )

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_loc = os.path.join(self.root_dir, idx[0])
        sample = load_board(self.game, sample_loc)

        if self.transform:
            sample = self.transform(sample)

        return sample
