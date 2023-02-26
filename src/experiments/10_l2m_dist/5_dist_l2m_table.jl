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

# Create an empty destination for each dataframe
mets = Dict{String, DataFrame}()

# Iterate over each permutation
for perm in readdir(perms_dir)
    # Point to the directory of the most recent metrics in this permutation
    full_perm = joinpath(perms_dir, perm)
    top_dir = readdir(full_perm, join=true)[end]
    # Iterate over every metric in this permutation
    for metric in readdir(top_dir)
        # Check that we have a dataframe for each metric
        if !haskey(mets, metric)
            # If we are missing a dataframe, initialize it with the correct columns
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
        # Create a new dataframe entry manually from the l2metrics in the file
        new_entry = (
            perm,
            md["perf_maintenance_mrlep"],
            md["forward_transfer_ratio"],
            md["backward_transfer_ratio"],
        )
        push!(mets[metric], new_entry)
    end

    # # Make a latex version of the dataframe and save
    # new_df_tex = latexify(new_df, env=:table, fmt="%.3f")
    # table_file = paper_results_dir("condensed.tex")
    # open(table_file, "w") do f
    #     write(f, new_df_tex)
    # end
end

# Point to the save directory
savedir(args...) = results_dir("processed", args...)
mkpath(savedir())

# Save the raw metrics
for (metric, df) in mets
    savefile = savedir(metric * ".csv")
    CSV.write(savefile, df)
end
