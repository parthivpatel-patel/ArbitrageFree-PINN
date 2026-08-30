# Arbitrage-Free Physics-Informed Neural Networks (PINN) for Quantitative Pricing & Volatility Surface Modeling

A high-performance quantitative finance engine combining **C++ systems architecture** with **PyTorch Physics-Informed Neural Networks (PINNs)** to model option pricing surfaces while strictly enforcing no-arbitrage constraints (calendar spread and butterfly spread monotonicity/convexity).

---

## Architecture & Tech Stack

* **High-Performance Core (C++ / Python C-Extensions):** Multithreaded tick parsers and analytical pricing engines optimized for low-latency market data processing.
* **Deep Learning Core (PyTorch):** Custom PINN loss functions enforcing structural no-arbitrage boundary conditions directly into the neural network's gradient descent optimization.
* **Alternative Data & NLP:** FinBERT-based sentiment and feature extraction pipelines parsing unstructured financial text.
* **Backtesting & Execution:** Historical and live event backtesting pipelines built for quantitative options strategies.

---

## Repository Structure

```text
├── src_cpp/                 # C++ multithreaded parsers, option pricing engines, and bindings
├── src_python/              # PyTorch PINN models, custom loss functions, and backtesters
├── data/                    # Market tick data and surface visualization artifacts
├── tests/                   # Pytest suite verifying numerical accuracy and structural constraints
└── setup.py                 # C++ extension compilation script