# rsn/prediction/inference.py

import torch
import numpy as np


def predict_signal(model, latest_state, mean, std):

    x = torch.tensor(latest_state, dtype=torch.float32).unsqueeze(0)

    mu, logvar = model(x)

    pred_return = mu.item()
    uncertainty = float(torch.exp(logvar).item())

    # --------------------------
    # Recover real price
    # --------------------------
    normalized_price = latest_state[0]
    current_price = normalized_price * std[0] + mean[0]

    # --------------------------
    # Direction
    # --------------------------
    side = "LONG" if pred_return > 0 else "SHORT"

    # --------------------------
    # Entry price（当前价）
    # --------------------------
    entry_price = current_price

    # --------------------------
    # Expected future price
    # --------------------------
    future_price = current_price * (1 + pred_return)

    # --------------------------
    # Volatility (用于风控)
    # --------------------------
    volatility = abs(latest_state[4])  # normalized volatility

    # --------------------------
    # Risk management
    # --------------------------
    sl_ratio = volatility * 1.5
    tp_ratio = volatility * 3

    if side == "LONG":
        stop_loss = entry_price * (1 - sl_ratio)
        take_profit = entry_price * (1 + tp_ratio)
    else:
        stop_loss = entry_price * (1 + sl_ratio)
        take_profit = entry_price * (1 - tp_ratio)

    # --------------------------
    # Confidence
    # --------------------------
    confidence = 1 / (1 + uncertainty)

    return {
        "side": side,
        "entry_price": float(entry_price),
        "expected_price": float(future_price),
        "expected_return": float(pred_return),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "confidence": float(confidence)
    }