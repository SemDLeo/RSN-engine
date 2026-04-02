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
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.net(x)