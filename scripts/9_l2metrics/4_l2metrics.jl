"""
    4_l2metrics.jl

# Description
Runs the l2metrics batch script from within Julia.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

using
    PyCall,
    Revise,
    DrWatson

# Experiment save directory name
experiment_top = "9_l2metrics"

# DCCR project files
include(projectdir("src", "setup.jl"))

# Get the location of the last log
last_log = readdir(results_dir("logs"))[end]

# Get the batch script location
exp_dir(args...) = projectdir("src", "experiments", experiment_top, args...)
l2m_script = "4_l2metrics.bat"
full_l2m_script = exp_dir(l2m_script)

# Run the command for the batch script
run(`cmd /c activate l2mmetrics \&\& $full_l2m_script $last_log`)
