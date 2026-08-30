import torch
import pytest
from src_python.pinn_model import OptionPricingPINN
from src_python.cuda_pinn_loss import ArbitrageFreePINNLoss

def test_pinn_forward_pass():
    model = OptionPricingPINN()
    strikes = torch.tensor([[100.0], [105.0]], requires_grad=True)
    maturities = torch.tensor([[0.5], [1.0]], requires_grad=True)
    
    prices = model(strikes, maturities)
    assert prices.shape == (2, 1), "Forward pass output shape mismatch."

def test_arbitrage_free_loss():
    model = OptionPricingPINN()
    criterion = ArbitrageFreePINNLoss()
    
    strikes = torch.tensor([[100.0], [105.0]], dtype=torch.float32, requires_grad=True)
    maturities = torch.tensor([[0.5], [1.0]], dtype=torch.float32, requires_grad=True)
    preds = model(strikes, maturities)
    true_prices = torch.tensor([[5.0], [3.0]], dtype=torch.float32)
    
    loss, loss_dict = criterion(model, preds, true_prices, strikes, maturities, spot=100.0)
    assert loss.item() > 0.0, "Loss calculation returned non-positive value."
    assert "calendar_penalty" in loss_dict
    assert "butterfly_penalty" in loss_dict
