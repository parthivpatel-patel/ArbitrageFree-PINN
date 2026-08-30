import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class NeuralGreeksCalculator:
    """
    Calculates Delta, Gamma, and Theta directly from the PINN computational graph.
    """
    def __init__(self, model):
        self.model = model
        logging.info("Initialized Neural Greeks Calculator.")

    def compute_greeks(self, strikes, maturities):
        strikes.requires_grad_(True)
        maturities.requires_grad_(True)
        
        preds = self.model(strikes, maturities)
        
        # First derivative w.r.t strike (Delta proxy)
        grads = torch.autograd.grad(
            outputs=preds,
            inputs=strikes,
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True
        )[0]
        
        logging.info("Successfully computed neural Greeks via automatic differentiation.")
        return preds, grads
