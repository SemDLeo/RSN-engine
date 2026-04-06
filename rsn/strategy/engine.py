# rsn/strategy/engine.py

import numpy as np


class StrategyEngine:
    """
    RSN Trading Strategy Engine

    Controls:
    - Trade filtering
    - Position decision
    - Risk management
    """

    def __init__(
        self,
        min_return=0.20,     # Minimum expected return (20%)
        min_confidence=0.65,  # Minimum confidence threshold (65%)
        cooldown_period=3,    # Cooldown period (5 time steps)
        risk_per_trade=0.05   # Risk per trade (5% of capital)
    ):
        self.min_return = min_return
        self.min_confidence = min_confidence
        self.cooldown_period = cooldown_period
        self.risk_per_trade = risk_per_trade

        self.cooldown = 0

    # -------------------------
    # Decide whether to trade
    # -------------------------
    def should_trade(self, signal):

        if self.cooldown > 0:
            return False

        if abs(signal["expected_return"]) < self.min_return:
            return False

        if signal["confidence"] < self.min_confidence:
            return False

        return True

    # -------------------------
    # Position sizing
    # -------------------------
    def compute_position_size(self, capital, entry_price, stop_loss):

        risk_amount = capital * self.risk_per_trade

        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0

        position_size = risk_amount / risk_per_unit

        return position_size

    # -------------------------
    # Generate trade decision
    # -------------------------
    def generate_trade(self, signal, capital):

        if not self.should_trade(signal):
            return None

        side = signal["side"]
        entry = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        take_profit = signal["take_profit"]

        size = self.compute_position_size(capital, entry, stop_loss)

        # set cooldown after generating a trade
        self.cooldown = self.cooldown_period

        return {
            "side": side,
            "entry_price": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": size,
            "confidence": signal["confidence"],
            "expected_return": signal["expected_return"]
        }

    # -------------------------
    # Update cooldown
    # -------------------------
    def step(self):
        if self.cooldown > 0:
            self.cooldown -= 1