import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class OptionsBacktester:
    """
    Event-driven backtesting harness for simulating delta-hedging and options strategies.
    """
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        logging.info("Initializing Options Backtester with capital: $" + format(initial_capital, ",.2f"))

    def run_backtest(self, pnl_stream):
        total_pnl = sum(pnl_stream)
        final_val = self.initial_capital + total_pnl
        return_pct = (total_pnl / self.initial_capital) * 100.0
        logging.info("Backtest Complete | Final Capital: $" + format(final_val, ",.2f") + " | Return: " + f"{return_pct:.2f}%")
        return final_val, return_pct

if __name__ == "__main__":
    bt = OptionsBacktester()
    bt.run_backtest([1200.0, -450.0, 3200.0, -800.0])
