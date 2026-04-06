# rsn/prediction/dataset.py

import pandas as pd
import numpy as np
import torch
from ta.momentum import RSIIndicator
from ta.trend import MACD


def load_btc_historical(path):

    df = pd.read_csv(path)

    # -------- Fix numeric --------
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    close = df["Close"]

    # -------- Feature Engineering --------
    df["RSI"] = RSIIndicator(close).rsi()
    df["MACD"] = MACD(close).macd()
    df["Return"] = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    df.dropna(inplace=True)

    # -------- Features --------
    features = df[[
        "Close", "RSI", "MACD",
        "Return", "Volatility", "Volume"
    ]].values

    # -------- Target (keypoint:predecting next return) --------
    future_return = df["Return"].shift(-1).dropna().values

    features = features[:-1]

    # -------- Normalize --------
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8

    features = (features - mean) / std

    np.save("feature_mean.npy", mean)
    np.save("feature_std.npy", std)

    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(future_return.reshape(-1, 1), dtype=torch.float32)

    print("X:", X.shape, "Y:", Y.shape)

    return X, Y