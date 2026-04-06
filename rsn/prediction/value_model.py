# rsn/prediction/value_model.py

import torch
import torch.nn as nn


class ValueModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)