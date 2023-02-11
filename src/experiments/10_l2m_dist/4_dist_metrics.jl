"""
    4_dist_metrics.jl

# Description
Runs the l2metrics on the latest logs from within Julia.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

using
    Combinatorics,
    PyCall,
    Revise,
    DrWatson

# Experiment save directory name
experiment_top = "10_l2m_dist"
log_dir_name = "logs"
metrics_dir_name = "l2metrics"
conda_env_name = "l2m"

# Declare all of the metrics being calculated
metrics = [
    "performance",
    "art_match",
    "art_activation",
]

# DCCR project files
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# GENERATE METRICS
# -----------------------------------------------------------------------------

# Get a list of the order indices
orders = collect(1:6)

# Create an iterator for all permutations and make it into a list
orders = collect(permutations(orders))

# Iterate over every one of the order folders
for order in orders
    # String of the permutation order
    text_order = String(join(order))

    # Get the most recent log directory name
    last_log = readdir(results_dir(log_dir_name, text_order))[end]

    # Set the full source directory
    src_dir = results_dir(log_dir_name, text_order, last_log)

    # Iterate over every metric
    for metric in metrics
        # Point to the output directory for this metric
        out_dir = results_dir(metrics_dir_name, text_order, last_log, metric)
        mkpath(out_dir)
        # Set the common python l2metrics command
        l2metrics_command = `python -m l2metrics --no-plot -p $metric -o $metric -O $out_dir -l $src_dir`
        if Sys.iswindows()
            run(`cmd /c activate $conda_env_name \&\& $l2metrics_command`)
        elseif Sys.isunix()
            run(`$l2metrics_command`)
        end
    end
end
