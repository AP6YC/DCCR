using
    PyCall,
    DrWatson

# DCCR project files
include(projectdir("src", "setup.jl"))

# Special folders for this experiment
include(projectdir("src", "setup_l2.jl"))
