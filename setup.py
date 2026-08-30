from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "quant_math",
        ["src_cpp/pricing_engine.cpp"],
        define_macros=[("VERSION_INFO", "1.0.0")],
    ),
]

setup(
    name="ArbitrageFreePINN",
    version="1.0.0",
    author="Parthiv S. Patel",
    description="Institutional-grade options pricing engine with PINNs and C++ backend",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pytest>=7.0.0"
    ],
)
