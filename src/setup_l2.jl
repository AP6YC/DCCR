"""
    setup_l2.jl

# Description
Special setup for l2logger and l2metrics experiments.
**NOTE** Must be run after setup.jl

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Point to the configs folder for the given experiment
configs_dir(args...) = projectdir("work", "configs", experiment_top, args...)

# Make the config folder for the experiment if it does not exist
mkpath(configs_dir())

# Include the common functions and structs for l2 experiments
include("lib_l2.jl")
