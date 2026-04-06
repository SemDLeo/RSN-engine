# examples/rsn_inference_test.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd

from rsn.prediction.transition_model import TransitionModel
from rsn.prediction.inference import predict_signal
from rsn.strategy.engine import StrategyEngine


# =========================
# Load model
# =========================
def load_model():
    model = TransitionModel()
    model.load_state_dict(torch.load("transition.pth"))
    model.eval()
    return model


# =========================
# Load normalization
# =========================
def load_norm():
    mean = np.load("feature_mean.npy")
    std = np.load("feature_std.npy")
    return mean, std


# =========================
# Prepare dataframe
# =========================
def prepare_df(path):

    df = pd.read_csv(path)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    df.dropna(inplace=True)

    return df


# =========================
# Convert row → state
# =========================
def row_to_state(row, mean, std):

    state = np.array([
        row["Close"],
        50,  # RSI placeholder
        0,   # MACD placeholder
        row["Return"],
        row["Volatility"],
        row["Volume"]
    ])

    return (state - mean) / std


# =========================
# Backtest Simulation
# =========================
def backtest(df, model, mean, std):

    strategy = StrategyEngine()

    capital = 10000
    trades = []

    for i in range(50, len(df) - 1):

        strategy.step()

        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        state = row_to_state(row, mean, std)

        signal = predict_signal(model, state, mean, std)

        trade = strategy.generate_trade(signal, capital)

        if trade is None:
            continue

        entry = trade["entry_price"]
        exit_price = next_row["Close"]

        # =========================
        # PnL calculation
        # =========================
        if trade["side"] == "LONG":
            pnl = (exit_price - entry) / entry
        else:
            pnl = (entry - exit_price) / entry

        capital *= (1 + pnl)
        trades.append(pnl)

    # =========================
    # Stats
    # =========================
    total_return = (capital - 10000) / 10000
    win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0

    print("\n===== BACKTEST RESULT =====")
    print(f"Final Capital: {capital:.2f}")
    print(f"Total Return: {total_return:.4f}")
    print(f"Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}")
    print("===========================\n")


# =========================
# Main
# =========================
def run():

    print("Loading model...")
    model = load_model()

    print("Loading normalization...")
    mean, std = load_norm()

    print("Loading data...")
    df = prepare_df("data/btc_15m.csv")

    print("Running backtest...\n")
    backtest(df, model, mean, std)


if __name__ == "__main__":
    run()