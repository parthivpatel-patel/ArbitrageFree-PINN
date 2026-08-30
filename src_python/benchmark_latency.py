import time
import torch
from pinn_model import OptionPricingPINN
import quant_math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_latency_benchmark():
    logging.info("Starting institutional latency benchmarking suite...")
    
    # 1. Benchmark C++ Analytical Pricing
    cpp_engine = quant_math.AnalyticalPricingEngine()
    start_time = time.perf_counter()
    for _ in range(10000):
        cpp_engine.black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)
    cpp_duration = (time.perf_counter() - start_time) * 1000.0 # ms
    logging.info(f"C++ Engine 10,000 evaluations: {cpp_duration:.2f} ms ({cpp_duration/10000:.4f} ms/call)")

    # 2. Benchmark PINN Forward Pass Inference
    model = OptionPricingPINN()
    model.eval()
    strikes = torch.tensor([[100.0]] * 1000)
    maturities = torch.tensor([[1.0]] * 1000)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(strikes, maturities)
    pinn_duration = (time.perf_counter() - start_time) * 1000.0 # ms
    logging.info(f"PINN Batch (1,000 options) 100 runs: {pinn_duration:.2f} ms ({pinn_duration/100:.2f} ms/batch)")

if __name__ == "__main__":
    run_latency_benchmark()
