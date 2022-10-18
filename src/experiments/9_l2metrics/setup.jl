"""
    setup.jl

Description:
    This script sets up the PyCall environment for the l2metrics experiment.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 9/28/2022: Created and documented.
"""

using Pkg
using Logging

# Set the correct Python environment and build PyCall
# ENV["PYTHON"] = raw"C:\Users\Sasha\Anaconda3\envs\dccr\python.exe"
# ENV["PYTHON"] = "C:\\Users\\Sasha\\Anaconda3\\envs\\dccr\\python.exe"
# ENV["PYTHON"] = "C:\\Users\\Sasha\\Anaconda3\\envs\\dccr"

# Load the default environment
ENV["PYTHON"] = ""
Pkg.build("PyCall")

# Load PyCall and Conda after setting the default environment
using PyCall
using Conda

# Use pip through conda because l2logger and l2metrics are not on conda/conda-forge
Conda.pip_interop(true)

# Install the l2loggers
Conda.pip("install", ["l2logger", "l2metrics"])
