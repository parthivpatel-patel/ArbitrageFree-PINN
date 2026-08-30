#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace py = pybind11;

class AnalyticalPricingEngine {
public:
    double black_scholes_call(double S, double K, double T, double r, double sigma) {
        if (T <= 0.0) return std::max(0.0, S - K);
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }

private:
    double norm_cdf(double value) {
        return 0.5 * std::erfc(-value * M_SQRT1_2);
    }
};

PYBIND11_MODULE(quant_math, m) {
    m.doc() = "High-performance C++ pricing and quantitative finance module";
    py::class_<AnalyticalPricingEngine>(m, "AnalyticalPricingEngine")
        .def(py::init<>())
        .def("black_scholes_call", &AnalyticalPricingEngine::black_scholes_call);
}
