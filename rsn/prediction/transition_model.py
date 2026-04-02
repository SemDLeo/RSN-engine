# rsn/prediction/transition_model.py

import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Mean (μ)
        self.mu_head = nn.Linear(64, input_dim)

        # Log variance (log σ²)
        self.logvar_head = nn.Linear(64, input_dim)

    def forward(self, x):
        h = self.shared(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar