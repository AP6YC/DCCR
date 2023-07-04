"""
    5_l2m_to_table.jl

# Description
Generates a LaTeX table from the l2 metrics from previous experiments.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""


using
    DrWatson,
    DataFrames,
    Revise,
    JSON,
    Latexify

experiment_top = "9_l2metrics"

# DCCR project files
include(projectdir("src", "setup.jl"))

# Select the metrics that we are interested in consolidating into a new dict
desired_metrics = [
    "perf_maintenance_mrlep",
    "forward_transfer_ratio",
    "backward_transfer_ratio",
]
pretty_rows = [
    "Activation",
    "Match",
    "Performance",
]

# Point to the directory of the most recent metrics
top_dir = readdir(results_dir("l2metrics"), join=true)[end]
# Create a new destination dictionary for the metrics of interest
new_metric_dict = Dict{String, Dict}()
# Iterate over every metric in the most recent metrics dir
dirs = readdir(top_dir)
# for dir in readdir(top_dir)
new_metric_array = zeros(3, length(desired_metrics))
for ix in eachindex(dirs)
    dir = dirs[ix]
    # Parse the metric JSON file and add the desired entry to the new dict
    metric_file = joinpath(top_dir, dir, dir * "_metrics.json")
    metric_dict = JSON.parsefile(metric_file)
    new_metric_dict[dir] = Dict{String, Any}(
        des_met => metric_dict[des_met] for des_met in desired_metrics
    )
    for jx in eachindex(desired_metrics)
        new_metric_array[ix, jx] = metric_dict[desired_metrics[jx]]
    end
end

# Make a dataframe with the array entries
new_df = DataFrame(
    Metric = pretty_rows,
    PM = new_metric_array[:, 1],
    FTR = new_metric_array[:, 2],
    BTR = new_metric_array[:, 3],
)

# Make a latex version of the dataframe and save
new_df_tex = latexify(new_df, env=:table, fmt="%.3f")
table_file = paper_results_dir("condensed.tex")
open(table_file, "w") do f
    write(f, new_df_tex)
end
