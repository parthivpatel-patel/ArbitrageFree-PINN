#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

double norm_cdf(double value) { return 0.5 * std::erfc(-value * M_SQRT1_2); }
double norm_pdf(double value) { return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * value * value); }

double bs_put(double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

double bs_vega(double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return S * norm_pdf(d1) * std::sqrt(T);
}

double calculate_implied_volatility(double target_price, double S, double K, double T, double r) {
    double sigma = 0.5;
    for (int i = 0; i < 100; ++i) {
        double price = bs_put(S, K, T, r, sigma);
        double diff = price - target_price;
        if (std::abs(diff) < 1e-5) return sigma;
        double vega = bs_vega(S, K, T, r, sigma);
        if (std::abs(vega) < 1e-8) return -1.0; 
        sigma = sigma - (diff / vega);
    }
    return -1.0;
}

PYBIND11_MODULE(quant_math, m) {
    m.doc() = "C++ High-Frequency Options Math Module";
    m.def("get_implied_volatility", &calculate_implied_volatility, "Calculates IV");
}
