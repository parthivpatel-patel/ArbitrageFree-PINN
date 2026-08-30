import torch
import torch.optim as optim
from pinn_model import OptionPricingPINN
from cuda_pinn_loss import ArbitrageFreePINNLoss
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_arbitrage_free_pinn():
    logging.info("Initializing Arbitrage-Free PINN Training Engine...")
    
    model = OptionPricingPINN()
    criterion = ArbitrageFreePINNLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Dummy collocation points for market ticks (Strike, Maturity)
    strikes = torch.linspace(80.0, 120.0, 100).unsqueeze(-1)
    maturities = torch.linspace(0.1, 2.0, 100).unsqueeze(-1)
    true_prices = torch.exp(-0.05 * maturities) * torch.relu(100.0 - strikes) + 5.0

    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        preds = model(strikes, maturities)
        loss, loss_dict = criterion(model, preds, true_prices, strikes, maturities, spot=100.0)
        
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {loss.item():.4f} | Data Loss: {loss_dict['data_loss']:.4f} | Calendar Penalty: {loss_dict['calendar_penalty']:.4f} | Butterfly Penalty: {loss_dict['butterfly_penalty']:.4f}")

    logging.info("PINN training and no-arbitrage surface calibration complete.")

if __name__ == "__main__":
    train_arbitrage_free_pinn()
