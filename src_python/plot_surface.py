import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pinn_model import ArbitrageFreeVolSurfacePINN

def generate_manuscript_plots():
    print("[PLOTTER] Generating high-resolution volatility surface mesh...")
    
    model = ArbitrageFreeVolSurfacePINN()
    model.eval()
    
    k = np.linspace(-0.25, 0.25, 50)
    tau = np.linspace(7.0/365.0, 90.0/365.0, 50)
    K, Tau = np.meshgrid(k, tau)
    
    k_tensor = torch.tensor(K.flatten(), dtype=torch.float32).unsqueeze(1)
    tau_tensor = torch.tensor(Tau.flatten(), dtype=torch.float32).unsqueeze(1)
    dummy_nlp = torch.zeros((k_tensor.shape[0], 768))
    
    with torch.no_grad():
        W = model(dummy_nlp, k_tensor, tau_tensor)
        IV = torch.sqrt(W / tau_tensor).numpy()
        IV = IV.reshape(K.shape)
        
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K, Tau * 365, IV * 100, cmap='viridis', edgecolor='none')
    
    ax.set_title("Arbitrage-Free Implied Volatility Surface (PINN)", fontsize=12)
    ax.set_xlabel("Log-Moneyness (k)", fontsize=10)
    ax.set_ylabel("Time to Maturity (Days)", fontsize=10)
    ax.set_zlabel("Implied Volatility (%)", fontsize=10)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    output_path = "../data/volatility_surface.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[PLOTTER] Publication figure successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_manuscript_plots()