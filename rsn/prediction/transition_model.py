# rsn/prediction/transition_model.py

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.mu = nn.Linear(64, 1)
        self.logvar = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)