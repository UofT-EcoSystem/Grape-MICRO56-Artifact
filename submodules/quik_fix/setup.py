from setuptools import find_packages, setup


setup(
    name="Quik-Fix",
    version="1.0",
    description="Understanding and Plotting Performance Metrics",
    author="Bojian Zheng",
    author_email="bojian@cs.toronto.edu",
    url="https://github.com/ArmageddonKnight/quik_fix",
    packages=find_packages(),
    scripts=[
        "scripts/alignown",
        "scripts/build_pytorch_here",
        "scripts/nsys/nsys_nvprof",
        "scripts/nsys/nsys_stats",
    ],
    install_requires=["cloudpickle", "matplotlib", "nvtx", "pybind11"],
)
