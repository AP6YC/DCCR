"""
    lib.jl

# Description
Aggregates all library code for the `DCCR` project.
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Exported constant for the version of the project
include("version.jl")

# C3 library code
include("c3/lib.jl")

# Lifelong learning library code
include("l2/lib.jl")
