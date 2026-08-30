import torch
import torch.nn as nn

class ArbitrageFreePINNLoss(nn.Module):
    """
    Institutional-grade Physics-Informed Neural Network loss function
    enforcing strict no-arbitrage boundary conditions for options pricing:
    1. Calendar Spread Arbitrage: C(T2, K) >= C(T1, K) for T2 > T1
    2. Butterfly Spread Arbitrage: d^2 C / dK^2 >= 0 (Convexity w.r.t strike)
    """
    def __init__(self, lambda_data=1.0, lambda_calendar=10.0, lambda_butterfly=10.0):
        super(ArbitrageFreePINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_data = lambda_data
        self.lambda_calendar = lambda_calendar
        self.lambda_butterfly = lambda_butterfly

    def forward(self, model, price_pred, price_true, strike, maturity, spot):
        # 1. Data Pricing Loss (MSE against market ticks)
        data_loss = self.mse(price_pred, price_true)

        # Enable gradient tracking for second-order derivatives
        strike.requires_grad_(True)
        maturity.requires_grad_(True)
        
        pred_prices = model(strike, maturity)

        # First derivative w.r.t maturity (Calendar spread constraint: dC/dT >= 0)
        grad_t = torch.autograd.grad(
            outputs=pred_prices, inputs=maturity,
            grad_outputs=torch.ones_like(pred_prices),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        calendar_penalty = torch.mean(torch.relu(-grad_t))

        # Second derivative w.r.t strike (Butterfly spread constraint: d2C/dK2 >= 0)
        grad_k = torch.autograd.grad(
            outputs=pred_prices, inputs=strike,
            grad_outputs=torch.ones_like(pred_prices),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_kk = torch.autograd.grad(
            outputs=grad_k, inputs=strike,
            grad_outputs=torch.ones_like(grad_k),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        butterfly_penalty = torch.mean(torch.relu(-grad_kk))

        # Total composite institutional loss
        total_loss = (
            self.lambda_data * data_loss +
            self.lambda_calendar * calendar_penalty +
            self.lambda_butterfly * butterfly_penalty
        )

        return total_loss, {
            "data_loss": data_loss.item(),
            "calendar_penalty": calendar_penalty.item(),
            "butterfly_penalty": butterfly_penalty.item()
        }
