"""A deliberately flawed model used as a review fixture.

Known issues (for evaluating recall):
1. unused import: os
2. `state` is passed to forward but unused (dead parameter)
3. dropout applied in eval without mode guard (train/eval issue)
4. no .to(device) handling anywhere (device issue)
"""


import torch
from torch import nn


class BadNet(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.5)
        self.state = None

    def forward(self, x, state=None):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)
