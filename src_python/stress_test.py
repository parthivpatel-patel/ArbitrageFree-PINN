import torch
from pinn_model import OptionPricingPINN
from cuda_pinn_loss import ArbitrageFreePINNLoss
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_stress_test():
    logging.info("Executing institutional extreme market shock stress test...")
    
    model = OptionPricingPINN()
    criterion = ArbitrageFreePINNLoss()
    
    # Simulate extreme shock conditions (e.g., 30% spot drop, massive IV expansion)
    stressed_strikes = torch.linspace(50.0, 150.0, 50, requires_grad=True).unsqueeze(-1)
    stressed_maturities = torch.linspace(0.01, 3.0, 50, requires_grad=True).unsqueeze(-1)
    
    preds = model(stressed_strikes, stressed_maturities)
    dummy_true = torch.zeros_like(preds)
    
    # Evaluate no-arbitrage penalty under stress
    _, loss_dict = criterion(model, preds, dummy_true, stressed_strikes, stressed_maturities, spot=70.0)
    
    logging.info(f"Stress Test Completed | Calendar Penalty: {loss_dict['calendar_penalty']:.6f} | Butterfly Penalty: {loss_dict['butterfly_penalty']:.6f}")
    assert loss_dict['calendar_penalty'] < 1e-3, "Calendar arbitrage boundary breached under stress."
    assert loss_dict['butterfly_penalty'] < 1e-3, "Butterfly convexity breached under stress."
    logging.info("All no-arbitrage invariants successfully held under extreme market shocks.")

if __name__ == "__main__":
    run_stress_test()
