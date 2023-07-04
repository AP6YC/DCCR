"""
    run_all.jl

# Description
This file runs all steps of the l2metrics experiments:
1. Set up the l2 packages in the Python environment.
2. Generate the scenario from the the structure of the data.
3. Run the l2 scenario with the DDVFA agent, logging the results.
4. Generate the metrics from the l2 logs.
"""

# -----------------------------------------------------------------------------
# SETUP
#
# Point to the Python environment that you want to use
# -----------------------------------------------------------------------------

# Default internal Python environment
# ENV["PYTHON"] = ""
# Custom existing Python environment
ENV["PYTHON"] = raw"C:\Users\Sasha\Anaconda3\envs\l2mmetrics\python.exe"

# -----------------------------------------------------------------------------
# RUN SUBEXPERIMENTS
# -----------------------------------------------------------------------------

# Setup the Python l2packages (i.e., l2logger and l2metrics)
include("1_setup_l2_packages.jl")

# Reverse-generate the scenario from the existing data
include("2_gen_scenario.jl")

# Run the l2 experiment and generate logs
include("3_driver.jl")

# Generate the metrics
include("4_l2metrics_julia.jl")
