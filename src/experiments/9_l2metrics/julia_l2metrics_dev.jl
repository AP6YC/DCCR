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

# Import L2M Python libraries
l2logger = pyimport("l2logger.l2logger")
l2metrics = pyimport("l2metrics")

# Create the data logger
data_logger = l2logger.DataLogger(
    "work/results/9_l2metrics/logs",
    "simple",
    Dict(
        "metrics_columns" => ["reward"],
        "log_format_version" => "1.0"
    ),
    Dict(
        "author" => "Sasha Petrenko",
        "complexity" => "1-low",
        "difficulty" => "2-medium",
        "scenario_type" => "custom",
    )
)

# Experiment save directory name
# experiment_top = "9_condensed"

# Run the common setup methods (data paths, etc.)
# include(projectdir("src", "setup.jl"))
