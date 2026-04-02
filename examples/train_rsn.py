# examples/train_rsn.py

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.transition_model import TransitionModel
from model.value_model import ValueModel 
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ==========================
# Models
# ==========================

class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class ValueModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ==========================
# Dataset
# ==========================

class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.data[idx+1], dtype=torch.float32)
        )


class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, data, horizon=10):
        self.data = data
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.horizon

    def __getitem__(self, idx):
        f_t = self.data[idx]
        price_now = self.data[idx][0]
        price_future = self.data[idx + self.horizon][0]

        value = (price_future - price_now) / price_now

        return (
            torch.tensor(f_t, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32)
        )


# ==========================
# Data Preparation
# ==========================

def load_data():

    df = yf.download("BTC-USD", start="2025-01-01", end="2026-01-01", interval="1d")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # indicators
    df["RSI"] = RSIIndicator(close).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd()

    df = df.dropna()

    # features
    features = df[["Close", "RSI", "MACD"]].values

    # normalization
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features


# ==========================
# Training
# ==========================

def train_transition(model, loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            pred = model(x)
            loss = ((pred - y)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Transition] Epoch {epoch} Loss: {total_loss:.4f}")


def train_value(model, loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            pred = model(x)
            loss = ((pred - y)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Value] Epoch {epoch} Loss: {total_loss:.4f}")


# ==========================
# Main
# ==========================

def main():

    print("Loading data...")
    data = load_data()

    print("Building datasets...")
    transition_dataset = TransitionDataset(data)
    value_dataset = ValueDataset(data)

    transition_loader = DataLoader(transition_dataset, batch_size=32, shuffle=True)
    value_loader = DataLoader(value_dataset, batch_size=32, shuffle=True)

    input_dim = data.shape[1]

    transition_model = TransitionModel(input_dim)
    value_model = ValueModel(input_dim)

    print("Training Transition Model...")
    train_transition(transition_model, transition_loader)

    print("Training Value Model...")
    train_value(value_model, value_loader)

    print("Training completed.")

    # test prediction
    sample = torch.tensor(data[-1], dtype=torch.float32)
    next_state = transition_model(sample)
    value = value_model(sample)

    print("\nSample Prediction:")
    print("Next state:", next_state.detach().numpy())
    print("Value:", value.item())
    
    # ==========================
    # Save models
    # ==========================

    torch.save(transition_model.state_dict(), "transition.pth")
    torch.save(value_model.state_dict(), "value.pth")

    print("Models saved.")


if __name__ == "__main__":
    main()