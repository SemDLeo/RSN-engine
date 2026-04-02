# rsn/prediction/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim

from rsn.prediction.transition_model import TransitionModel
from rsn.prediction.value_model import ValueModel
from rsn.prediction.dataset import load_btc_dataset


# =========================
# Gaussian Negative Log Likelihood
# =========================
def gaussian_nll(mu, logvar, target):
    """
    mu: predicted mean
    logvar: predicted log variance
    target: ground truth
    """
    return torch.mean(
        0.5 * logvar + (target - mu) ** 2 / (2 * torch.exp(logvar) + 1e-8)
    )


# =========================
# Training Function
# =========================
def train(data_path, epochs=30, lr=1e-3):

    print("Loading dataset...")
    X, Y = load_btc_dataset(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = X.to(device)
    Y = Y.to(device)

    print("Device:", device)
    print("Dataset size:", X.shape)

    # =========================
    # Models
    # =========================
    transition_model = TransitionModel().to(device)
    value_model = ValueModel().to(device)

    # =========================
    # Optimizer
    # =========================
    optimizer = optim.Adam(
        list(transition_model.parameters()) +
        list(value_model.parameters()),
        lr=lr
    )

    mse_loss = nn.MSELoss()

    # =========================
    # Training Loop
    # =========================
    for epoch in range(epochs):

        optimizer.zero_grad()

        # ---- Transition Model ----
        mu, logvar = transition_model(X)

        # Important: limit logvar to prevent extreme values (stability)
        logvar = torch.clamp(logvar, min=-10, max=10)

        loss_transition = gaussian_nll(mu, logvar, Y)

        # ---- Value Model ----
        value_pred = value_model(X).squeeze()

        # use the future price (Y[:, 0]) as the target for value model
        loss_value = mse_loss(value_pred, Y[:, 0])

        # ---- Total Loss ----
        loss = loss_transition + loss_value

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(transition_model.parameters()) +
            list(value_model.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        # =========================
        # Logging
        # =========================
        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Total Loss: {loss.item():.6f} | "
                f"Transition: {loss_transition.item():.6f} | "
                f"Value: {loss_value.item():.6f}"
            )

    # =========================
    # Save Models
    # =========================
    torch.save(transition_model.state_dict(), "transition.pth")
    torch.save(value_model.state_dict(), "value.pth")

    print("✅ Models saved:")
    print(" - transition.pth")
    print(" - value.pth")