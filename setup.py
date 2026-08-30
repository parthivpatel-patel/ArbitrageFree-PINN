from setuptools import setup, Extension
import pybind11

# Define the C++ extension module
ext_modules = [
    Extension(
        "quant_math",                      # The name Python will use to import it
        ["src_cpp/pricing_engine.cpp"],    # The C++ file we wrote
        include_dirs=[pybind11.get_include()],
        language="c++"
    )
]

setup(
    name="quant_math",
    ext_modules=ext_modules,
)