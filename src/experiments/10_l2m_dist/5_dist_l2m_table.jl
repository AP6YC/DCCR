"""
    5_dist_metrics.jl

# Description
Collects the l2metrics into a table with statistics.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

using
    CSV,
    DrWatson,
    DataFrames,
    Revise,
    JSON,
    Latexify

experiment_top = "10_l2m_dist"

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

# Point to the results directory containing all of the permutations
perms_dir = results_dir("l2metrics")
mets = Dict{String, DataFrame}()
# mets = Dict{String, Dict}()
# Iterate over each permutation
# for perm in readdir(perms_dir, join=true)
for perm in readdir(perms_dir)
    # Point to the directory of the most recent metrics in this permutation
    full_perm = joinpath(perms_dir, perm)
    top_dir = readdir(full_perm, join=true)[end]
    # @info top_dir
    # Iterate over every metric in this permutation
    for metric in readdir(top_dir)
        # Check that we have a dataframe for each metric
        if !haskey(mets, metric)
            mets[metric] = DataFrame(
                perm=String[],
                pm=Float[],
                ftr=Float[],
                btr=Float[],
            )
        end
        # Load the metric file
        metric_file = joinpath(top_dir, metric, metric * "_metrics.json")
        md = JSON.parsefile(metric_file)
        # @info md["perf_maintenance_mrlep"]
        new_entry = (
            perm,
            md["perf_maintenance_mrlep"],
            md["forward_transfer_ratio"],
            md["backward_transfer_ratio"],
        )
        push!(mets[metric], new_entry)
    end
    # # Create a new destination dictionary for the metrics of interest
    # new_metric_dict = Dict{String, Dict}()
    # # Iterate over every metric in the most recent metrics dir
    # dirs = readdir(top_dir)
    # # for dir in readdir(top_dir)
    # new_metric_array = zeros(3, length(desired_metrics))
    # for ix in eachindex(dirs)
    #     dir = dirs[ix]
    #     # Parse the metric JSON file and add the desired entry to the new dict
    #     metric_file = joinpath(top_dir, dir, dir * "_metrics.json")
    #     metric_dict = JSON.parsefile(metric_file)
    #     new_metric_dict[dir] = Dict{String, Any}(
    #         des_met => metric_dict[des_met] for des_met in desired_metrics
    #     )
    #     for jx in eachindex(desired_metrics)
    #         new_metric_array[ix, jx] = metric_dict[desired_metrics[jx]]
    #     end
    # end

    # # Make a dataframe with the array entries
    # new_df = DataFrame(
    #     Metric = pretty_rows,
    #     PM = new_metric_array[:, 1],
    #     FTR = new_metric_array[:, 2],
    #     BTR = new_metric_array[:, 3],
    # )

    # # Make a latex version of the dataframe and save
    # new_df_tex = latexify(new_df, env=:table, fmt="%.3f")
    # table_file = paper_results_dir("condensed.tex")
    # open(table_file, "w") do f
    #     write(f, new_df_tex)
    # end

end

# Point to the save directory
savedir(args...) = results_dir("processed", args...)
# packingdir(args...) = packed_dir(experiment_top)

mkpath(savedir())
# mkpath(packingdir())

# Save the raw metrics
for (metric, df) in mets
    savefile = savedir(metric * ".csv")
    CSV.write(savefile, df)
end
