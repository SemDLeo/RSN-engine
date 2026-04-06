# rsn/prediction/online_trainer.py

import torch


class OnlineTrainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.last_X = None

    def step(self, current_X, next_return):

        if self.last_X is None:
            self.last_X = current_X
            return

        X = self.last_X
        Y = torch.tensor([[next_return]], dtype=torch.float32)

        mu, logvar = self.model(X)

        loss = ((mu - Y) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_X = current_X