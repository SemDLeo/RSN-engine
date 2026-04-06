# rsn/prediction/trainer.py

import torch
import torch.optim as optim

from rsn.prediction.transition_model import TransitionModel
from rsn.prediction.value_model import ValueModel
from rsn.prediction.dataset import load_btc_historical


def gaussian_nll(mu, logvar, y):
    return torch.mean(
        0.5 * logvar + (y - mu) ** 2 / (2 * torch.exp(logvar) + 1e-8)
    )


def train(path, epochs=20, lr=1e-3):

    print("Loading dataset...")
    X, Y = load_btc_historical(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = X.to(device)
    Y = Y.to(device)

    model = TransitionModel().to(device)
    value_model = ValueModel().to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(value_model.parameters()),
        lr=lr
    )

    for epoch in range(epochs):

        optimizer.zero_grad()

        mu, logvar = model(X)

        logvar = torch.clamp(logvar, -10, 10)

        loss1 = gaussian_nll(mu, logvar, Y)

        value_pred = value_model(X)
        loss2 = torch.mean((value_pred - Y) ** 2)

        loss = loss1 + 0.1 * loss2

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(value_model.parameters()),
            1.0
        )

        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "transition.pth")
    torch.save(value_model.state_dict(), "value.pth")

    print("✅ Models saved.")