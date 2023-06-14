"""
    serve.jl

# Description
Convenience script that serves the locally built documentation.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using LiveServer

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Make the documentation
include("make.jl")

# -----------------------------------------------------------------------------
# SERVE
# -----------------------------------------------------------------------------

# Serve the documentation for development
serve(dir="build")
