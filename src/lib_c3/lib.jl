"""
    lib.jl

# Description
Aggregates all of the C3 library code.
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Custom colors definitions
include("colors.jl")

# Data structures definitions
include("data.jl")

# Data and experiment utilities
include("utils.jl")

# Constants used throughout experiments
include("constants.jl")

# Myriad of experiment and plotting functions
include("functions.jl")
