# 9_l2metrics

This folder contains the experiment files necessary to generate L2 metrics for the project.

## Files

The `run_all.jl` script runs each step of the L2 logging and metrics experiments.
More granularly, these steps are:

1. `1_setup_l2_packages.jl`: sets up the PyCall environment, installing l2logging and l2metrics in a Python environment and building PyCall around it for later steps.
2. `2_gen_scenario.jl`: generates the scenario files from the structure of the existing dataset.
3. `3_driver.jl`: runs the full L2 experiment, logging with the PyCall environment that was set up in the first step.
4. `4_l2metrics_julia.jl`: runs the l2metrics command in the shell for each metric used in the project.
