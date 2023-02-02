using
    DrWatson,
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
# Point to the directory of the most recent metrics
top_dir = readdir(results_dir("l2metrics"), join=true)[end]
# Create a new destination dictionary for the metrics of interest
new_metric_dict = Dict{String, Dict}()
# Iterate over every metric in the most recent metrics dir
for dir in readdir(top_dir)
    # Parse the metric JSON file and add the desired entry to the new dict
    metric_file = joinpath(top_dir, dir, dir * "_metrics.json")
    metric_dict = JSON.parsefile(metric_file)
    new_metric_dict[dir] = Dict{String, Any}(
        des_met => metric_dict[des_met] for des_met in desired_metrics
    )
end
println(new_metric_dict)