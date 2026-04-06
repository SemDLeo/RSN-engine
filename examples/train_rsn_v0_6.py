# examples/train_rsn_v0_5.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import pandas as pd

from rsn.prediction.trainer import train
from binance.client import Client


# -------------------------
# Config
# -------------------------
DATA_PATH = "data/btc_15m.csv"
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE


# -------------------------
# Binance Client (public)
# -------------------------
client = Client()


# -------------------------
# Incremental Download
# -------------------------
def download_klines(symbol, interval, start_ts, end_ts):

    all_data = []

    while start_ts < end_ts:
        print(f"Fetching from {start_ts} ...")

        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=1000
        )

        if not klines:
            break

        all_data.extend(klines)

        # move forward
        start_ts = klines[-1][0] + 1

        time.sleep(0.3)  # avoid rate limit

    return all_data


# -------------------------
# Get 15m data (with cache)
# -------------------------
def get_15m_klines():

    if os.path.exists(DATA_PATH):
        print(f"✅ Using existing dataset: {DATA_PATH}")
        return DATA_PATH

    print("Downloading BTCUSDT 15m data from Binance...")

    start_str = "1 Jan, 2017"  # ⚠️ Binance才有数据
    end_str = "1 Apr, 2026"

    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)

    raw = download_klines(SYMBOL, INTERVAL, start_ts, end_ts)

    df = pd.DataFrame(raw, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Num_trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore"
    ])

    # convert numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    df = df[["Open_time", "Open", "High", "Low", "Close", "Volume"]]

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"✅ Download complete: {df.shape}")

    return DATA_PATH


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    data_path = get_15m_klines()

    print("\n🚀 Start training...\n")

    train(data_path, epochs=20, lr=1e-3)