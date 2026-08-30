import torch
import torch.nn as nn

class OptionPricingPINN(nn.Module):
    """
    Multilayer Perceptron (MLP) mapping option strikes and maturities 
    to theoretical option prices with smooth SiLU activations for high-order gradient stability.
    """
    def __init__(self, hidden_dim=128, num_layers=4):
        super(OptionPricingPINN, self).__init__()
        
        layers = [
            nn.Linear(2, hidden_dim),
            nn.SiLU()
        ]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, strike, maturity):
        # Concatenate strike and maturity as input feature tensor [N, 2]
        inputs = torch.cat([strike, maturity], dim=-1)
        return self.network(inputs)
