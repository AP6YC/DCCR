"""
    4_l2metrics.jl

# Description
Runs the l2metrics on the latest logs from within Julia.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

using
    PyCall,
    Revise,
    DrWatson

# Experiment save directory name
experiment_top = "9_l2metrics"

# DCCR project files
include(projectdir("src", "setup.jl"))

# Special folders for this experiment
# include(projectdir("src", "setup_l2.jl"))

# # Setup the PyCall environment
# include(projectdir("src", "setup_pycall_env.jl"))
last_log = readdir(results_dir("logs"))[end]

# metrics = [
#     "performance",
#     "art_match",
#     "art_activation",
# ]

exp_dir(args...) = projectdir("src", "experiments", experiment_top, args...)
l2m_script = "4_l2metrics.bat"
full_l2m_script = exp_dir(l2m_script)

run(`cmd /c activate l2mmetrics \&\& $full_l2m_script $last_log`)
# run(`cmd /c activate l2mmetrics \&\& dir $()`)
