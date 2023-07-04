"""
    lib.jl

# Description
Aggregates all of the lib_l2 source files.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Common definitions and variables
include("common.jl")

# Experience definitions
include("experience.jl")

# Scenario and experience definitions
include("scenario.jl")

# Agent definitions that depend on scenario definitions
include("agents.jl")
