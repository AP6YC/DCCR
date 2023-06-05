"""
    build_pyenv.jl

# Description
Sets and builds the PyCall environment for Julia experiments requiring Python modules.
"""

# using PyCall
using Pkg

# Point to the correct Python environment
include("set_pyenv.jl")

# Build the PyCall environment
Pkg.build("PyCall")

# Load PyCall and Conda after setting the default environment
using PyCall
