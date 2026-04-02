# examples/train_rsn.py

from rsn.prediction.trainer import train

if __name__ == "__main__":
    train("data/btc_usd.csv")