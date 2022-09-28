"""
    9_l2metrics.jl

Description:
    This script runs a single condensed scenario for logging.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 9/28/2022: Created and documented.
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Pkg
using Logging
# using PyCall

# Set the correct Python environment and build PyCall
# ENV["PYTHON"] = raw"C:\Users\Sasha\Anaconda3\envs\dccr\python.exe"
# Pkg.build("PyCall")
using PyCall

@info PyCall.pyversion

l2logger = pyimport("l2logger")
l2metrics = pyimport("l2metrics")

@info l2logger
@info l2metrics

# Experiment save directory name
# experiment_top = "9_condensed"

# Run the common setup methods (data paths, etc.)
# include(projectdir("src", "setup.jl"))
