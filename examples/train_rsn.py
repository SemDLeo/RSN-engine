# examples/train_rsn.py
import sys 
import os     
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rsn.prediction.trainer import train
import yfinance as yf
import pandas as pd

def download_btc():
    df = yf.download("BTC-USD", start="2010-01-01", interval="1d")
    df.to_csv("data/btc_usd.csv")
    print("BTC data downloaded.")

if __name__ == "__main__":
    download_btc()

    from rsn.prediction.trainer import train
    train("data/btc_usd.csv")

