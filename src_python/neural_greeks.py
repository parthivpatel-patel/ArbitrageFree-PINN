import torch

class NeuralGreeksEngine:
    """
    High-performance analytical Greeks calculator leveraging PyTorch automatic 
    differentiation on the trained PINN volatility surface model.
    """
    def __init__(self, model):
        self.model = model

    def compute_greeks(self, strike, maturity):
        strike.requires_grad_(True)
        maturity.requires_grad_(True)
        
        price = self.model(strike, maturity)
        
        # Delta: First derivative w.r.t strike / spot proxy
        delta = torch.autograd.grad(
            outputs=price, inputs=strike,
            grad_outputs=torch.ones_like(price),
            create_graph=True, retain_graph=True
        )[0]
        
        # Gamma: Second derivative w.r.t strike
        gamma = torch.autograd.grad(
            outputs=delta, inputs=strike,
            grad_outputs=torch.ones_like(delta),
            create_graph=True, retain_graph=True
        )[0]

        # Theta: First derivative w.r.t maturity (time decay)
        theta = torch.autograd.grad(
            outputs=price, inputs=maturity,
            grad_outputs=torch.ones_like(price),
            create_graph=True, retain_graph=True
        )[0]

        return {
            "price": price.detach().cpu().numpy(),
            "delta": delta.detach().cpu().numpy(),
            "gamma": gamma.detach().cpu().numpy(),
            "theta": theta.detach().cpu().numpy()
        }
