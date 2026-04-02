# rsn/prediction/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim

from rsn.prediction.transition_model import TransitionModel
from rsn.prediction.value_model import ValueModel
from rsn.prediction.dataset import load_btc_dataset


def train(data_path):

    X, Y = load_btc_dataset(data_path)

    transition_model = TransitionModel()
    value_model = ValueModel()

    optimizer = optim.Adam(
        list(transition_model.parameters()) +
        list(value_model.parameters()),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()

    for epoch in range(20):
        optimizer.zero_grad()

        pred_next = transition_model(X)
        pred_value = value_model(X).squeeze()

        loss1 = loss_fn(pred_next, Y)
        loss2 = loss_fn(pred_value, Y[:, 0])

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    torch.save(transition_model.state_dict(), "transition.pth")
    torch.save(value_model.state_dict(), "value.pth")

    print("Models saved.")