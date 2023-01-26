# using PyCall
using Pkg

# Point to the correct Python environment
include("set_pyenv.jl")

# Build the PyCall environment
Pkg.build("PyCall")

# Load PyCall and Conda after setting the default environment
using PyCall
