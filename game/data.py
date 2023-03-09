"""
StateDataset
"""

import torch


class StateDataset(torch.utils.data.Dataset):
    """
    StateDataset class
    """

    def __init__(self, samples, targets) -> None:
        assert len(samples) == len(targets)
        super().__init__()
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]
