"""
    setup_l2_packages.jl

# Description
This script sets up the PyCall environment for the l2metrics experiment.

# Authors
- Sasha Petrenko <sap625@mst.edu>

# Timeline
- 9/28/2022: Created and documented.
"""

# Setup the project environment
using DrWatson

# Setup the PyCall environment
include(projectdir("src", "setup_pycall_env.jl"))

# Load Conda after setting and building the PyCall environment
using Conda

# Use pip through conda because l2logger and l2metrics are not on conda/conda-forge
Conda.pip_interop(true)

# Install the l2loggers
Conda.pip("install", ["l2logger", "l2metrics"])
