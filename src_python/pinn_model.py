import torch
import torch.nn as nn

class ArbitrageFreeVolSurfacePINN(nn.Module):
    def __init__(self, nlp_embedding_dim=768, hidden_dim=256):
        super(ArbitrageFreeVolSurfacePINN, self).__init__()
        # Mish activation functions handle non-linear higher-order derivatives better than ReLU
        self.net = nn.Sequential(
            nn.Linear(nlp_embedding_dim + 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # mathematically guarantees total variance is strictly positive
        )

    def forward(self, x_nlp, k, tau):
        # Concatenate the NLP sentiment vector, log-moneyness (k), and time-to-maturity (tau)
        inputs = torch.cat([x_nlp, k, tau], dim=1)
        return self.net(inputs)

def compute_pinn_loss(model, x_nlp, k, tau, w_true, lambda_cal=100.0, lambda_but=100.0):
    """
    Computes the Mean Squared Error and the mathematical penalty for arbitrage violations.
    """
    # Enable gradient tracking to compute partial differential equations (PDEs)
    k.requires_grad_(True)
    tau.requires_grad_(True)
    
    # 1. Forward Pass
    w_hat = model(x_nlp, k, tau)
    mse_loss = torch.mean((w_hat - w_true)**2)
    
    # 2. Compute Gradients via Autograd
    dw_dtau = torch.autograd.grad(w_hat, tau, grad_outputs=torch.ones_like(w_hat), create_graph=True)[0]
    dw_dk = torch.autograd.grad(w_hat, k, grad_outputs=torch.ones_like(w_hat), create_graph=True)[0]
    d2w_dk2 = torch.autograd.grad(dw_dk, k, grad_outputs=torch.ones_like(dw_dk), create_graph=True)[0]
    
    # 3. Calendar Arbitrage Penalty (Variance must increase with time: dw/dtau >= 0)
    calendar_penalty = torch.mean(torch.relu(-dw_dtau)**2)
    
    # 4. Butterfly Arbitrage Penalty (Risk-neutral density must be positive)
    term1 = (1.0 - (k / (2.0 * w_hat)) * dw_dk)**2
    term2 = 0.25 * (dw_dk**2) * ((1.0 / w_hat) + 0.25)
    term3 = 0.5 * d2w_dk2
    g_k = term1 - term2 + term3
    butterfly_penalty = torch.mean(torch.relu(-g_k)**2)
    
    # Total Objective
    total_loss = mse_loss + (lambda_cal * calendar_penalty) + (lambda_but * butterfly_penalty)
    return total_loss, mse_loss, calendar_penalty, butterfly_penalty

if __name__ == "__main__":
    print("PINN Architecture Loaded Successfully.")
    print("Ready for data ingestion and calibration.")