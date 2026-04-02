# rsn/prediction/dataset.py

import pandas as pd
import numpy as np
import torch
from ta.momentum import RSIIndicator
from ta.trend import MACD

def load_btc_dataset(path, feature_dim=6):

    df = pd.read_csv(path)

    # =========================
    # Convert to numeric
    # =========================
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    close = df["Close"]

    # =========================
    # Feature Engineering
    # =========================
    df["RSI"] = RSIIndicator(close).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd()

    df["Return"] = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    df = df.dropna()

    # =========================
    # Feature Selection
    # =========================
    features = df[[
        "Close",
        "RSI",
        "MACD",
        "Return",
        "Volatility",
        "Volume"
    ]].values

    # =========================
    # Normalization (CRITICAL)
    # =========================
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8

    features = (features - mean) / std

    # 保存（用于推理）
    np.save("feature_mean.npy", mean)
    np.save("feature_std.npy", std)

    # =========================
    # Build Training Data
    # =========================
    X = torch.tensor(features[:-1], dtype=torch.float32)
    Y = torch.tensor(features[1:], dtype=torch.float32)

    print("Data types:")
    print(df.dtypes)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y