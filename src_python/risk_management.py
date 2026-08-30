import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class PortfolioRiskEngine:
    """
    Portfolio-level Value-at-Risk (VaR), Expected Shortfall (ES), and 
    stress testing calculator for multi-leg options books.
    """
    def __init__(self, confidence_level=0.99):
        self.confidence_level = confidence_level
        logging.info(f"Initializing Portfolio Risk Engine at {confidence_level*100}% confidence level...")

    def calculate_var(self, portfolio_pnl_stream):
        if len(portfolio_pnl_stream) == 0:
            return 0.0
        var = np.percentile(portfolio_pnl_stream, (1.0 - self.confidence_level) * 100.0)
        logging.info(f"Calculated Value-at-Risk (VaR): ")
        return abs(var)

    def calculate_expected_shortfall(self, portfolio_pnl_stream):
        var = self.calculate_var(portfolio_pnl_stream)
        tail_losses = [pnl for pnl in portfolio_pnl_stream if pnl <= -var]
        es = np.mean(tail_losses) if tail_losses else var
        logging.info(f"Calculated Expected Shortfall (ES): ")
        return abs(es)
