import torch
import pytest
from src_python.pinn_model import ArbitrageFreeVolSurfacePINN, compute_pinn_loss

def test_model_output_shape():
    model = ArbitrageFreeVolSurfacePINN(nlp_embedding_dim=768, hidden_dim=256)
    model.eval()
    
    batch_size = 10
    x_nlp = torch.zeros((batch_size, 768))
    k = torch.linspace(-0.1, 0.1, batch_size).unsqueeze(1)
    tau = torch.full((batch_size, 1), 30.0 / 365.0)
    
    with torch.no_grad():
        w = model(x_nlp, k, tau)
        
    assert w.shape == (batch_size, 1), "Output tensor shape mismatch."
    assert torch.all(w >= 0), "Total variance surface violates non-negativity bounds."

def test_pinn_loss_computation():
    model = ArbitrageFreeVolSurfacePINN(nlp_embedding_dim=768, hidden_dim=256)
    batch_size = 20
    x_nlp = torch.zeros((batch_size, 768), requires_grad=True)
    k = torch.linspace(-0.2, 0.2, batch_size).unsqueeze(1)
    tau = torch.full((batch_size, 1), 30.0 / 365.0)
    w_true = torch.full((batch_size, 1), 0.04)
    
    total_loss, mse, cal_pen, but_pen = compute_pinn_loss(
        model, x_nlp, k, tau, w_true, lambda_cal=10.0, lambda_but=10.0
    )
    
    assert not torch.isnan(total_loss), "Loss computed as NaN."
    assert total_loss.item() >= 0.0, "Loss cannot be negative."