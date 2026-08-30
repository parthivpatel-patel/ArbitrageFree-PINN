import pytest
import quant_math

def test_cpp_black_scholes_pricing():
    engine = quant_math.AnalyticalPricingEngine()
    
    # At-the-money call option test case
    price = engine.black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)
    assert price > 0.0, "Black-Scholes C++ pricing returned non-positive value."
    assert abs(price - 10.4505) < 1e-3, "Black-Scholes pricing deviates from expected baseline."

def test_cpp_intrinsic_pricing():
    engine = quant_math.AnalyticalPricingEngine()
    
    # Zero maturity intrinsic test case
    price = engine.black_scholes_call(110.0, 100.0, 0.0, 0.05, 0.2)
    assert price == 10.0, "Zero-maturity expiration pricing failed."
