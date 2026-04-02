# examples/rsn_inference.py
import sys   
import os     
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd

from rsn.core.node import RSNNode
from rsn.prediction.transition_model import TransitionModel
from rsn.prediction.value_model import ValueModel
from rsn.prediction.tree import build_tree


# =========================
# Load latest BTC state
# =========================
def load_latest_state(csv_path):

    df = pd.read_csv(csv_path)

    # Ensure numeric
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # ===== Rebuild features（必须和训练一致）=====
    from ta.momentum import RSIIndicator
    from ta.trend import MACD

    close = df["Close"]

    df["RSI"] = RSIIndicator(close).rsi()
    macd = MACD(close)
    df["MACD"] = macd.macd()

    df["Return"] = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    df = df.dropna()

    latest = df.tail(1)

    features = latest[[
        "Close",
        "RSI",
        "MACD",
        "Return",
        "Volatility",
        "Volume"
    ]].values[0]

    return features


# =========================
# Print tree (simple)
# =========================
def print_tree(node, max_depth=3):

    indent = "  " * node.depth
    print(f"{indent}- Depth {node.depth}, Value: {node.value}")

    if node.depth >= max_depth:
        return

    for child in node.children:
        print_tree(child, max_depth)


# =========================
# Main Inference
# =========================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")

    # =========================
    # Load models
    # =========================
    transition_model = TransitionModel().to(device)
    value_model = ValueModel().to(device)

    transition_model.load_state_dict(torch.load("transition.pth", map_location=device))
    value_model.load_state_dict(torch.load("value.pth", map_location=device))

    transition_model.eval()
    value_model.eval()

    # =========================
    # Load normalization params
    # =========================
    mean = np.load("feature_mean.npy")
    std = np.load("feature_std.npy")

    # =========================
    # Load initial state
    # =========================
    print("Loading latest BTC state...")
    initial_state = load_latest_state("data/btc_usd.csv")

    # Normalize
    initial_state = (initial_state - mean) / std

    print("Initial state:", initial_state)

    # =========================
    # Build RSN Tree
    # =========================
    root = RSNNode(initial_state)

    print("Building RSN tree...")

    build_tree(
        root,
        transition_model,
        value_model,
        depth_limit=4   # 未来4步
    )

    # =========================
    # Output results
    # =========================
    print("\nGenerated Future Tree:\n")
    print_tree(root, max_depth=3)


if __name__ == "__main__":
    main()