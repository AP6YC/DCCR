"""
    lib_l2.jl

# Description
Includes all of the lib_l2 source files.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    DataStructures,     # Dequeue
    PyCall,             # PyObject
    JSON                # JSON file load/save

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
