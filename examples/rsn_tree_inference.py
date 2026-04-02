# examples/rsn_tree_inference.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD

from model.transition_model import TransitionModel
from model.value_model import ValueModel


# ==========================
# Node
# ==========================

class RSNNode:
    def __init__(self, features, depth=0, parent=None):
        self.features = features
        self.depth = depth
        self.parent = parent
        self.children = []
        self.value = None


# ==========================
# Load data (same as training)
# ==========================

def load_latest_state():

    df = yf.download("BTC-USD", period="60d", interval="1d")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df["RSI"] = RSIIndicator(close).rsi()
    macd = MACD(close)
    df["MACD"] = macd.macd()

    df = df.dropna()

    features = df[["Close", "RSI", "MACD"]].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features[-1]  # latest state


# ==========================
# Tree Expansion
# ==========================

def expand_node(node, transition_model, value_model, branch_factor=3):

    for _ in range(branch_factor):

        f = torch.tensor(node.features, dtype=torch.float32)

        # predict next state
        next_f = transition_model(f).detach().numpy()

        child = RSNNode(next_f, depth=node.depth + 1, parent=node)

        # evaluate
        value = value_model(torch.tensor(next_f, dtype=torch.float32)).item()
        child.value = value

        node.children.append(child)


# ==========================
# Recursive Tree
# ==========================

def build_tree(root, transition_model, value_model, depth_limit=3):

    if root.depth >= depth_limit:
        return

    expand_node(root, transition_model, value_model)

    for child in root.children:
        build_tree(child, transition_model, value_model, depth_limit)


# ==========================
# Path Evaluation
# ==========================

def collect_paths(node, path=None, paths=None):

    if path is None:
        path = []
    if paths is None:
        paths = []

    path = path + [node]

    if not node.children:
        paths.append(path)
        return paths

    for child in node.children:
        collect_paths(child, path, paths)

    return paths


def evaluate_path(path):

    return sum(n.value for n in path if n.value is not None)


# ==========================
# Main
# ==========================

def main():

    print("Loading models...")

    input_dim = 3
    transition_model = TransitionModel(input_dim)
    value_model = ValueModel(input_dim)

    transition_model.load_state_dict(torch.load("transition.pth"))
    value_model.load_state_dict(torch.load("value.pth"))
    
    transition_model.eval()
    value_model.eval()

    print("Loading latest state...")
    state = load_latest_state()

    root = RSNNode(state)

    print("Building tree...")
    build_tree(root, transition_model, value_model, depth_limit=3)

    print("Evaluating paths...")
    paths = collect_paths(root)

    best_score = -1e9
    best_path = None

    for path in paths:
        score = evaluate_path(path)
        if score > best_score:
            best_score = score
            best_path = path

    print("\nBest Path:")
    for node in best_path:
        print(f"Depth {node.depth}, Value {node.value}")

    print("\nBest Score:", best_score)


if __name__ == "__main__":
    main()