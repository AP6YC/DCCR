"""
    6_stats.jl

# Description
Generates the statistics for all permutations of the generated l2metrics.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

using
    CSV,
    DrWatson,
    DataFrames,
    Latexify,
    Statistics

experiment_top = "10_l2m_dist"

# DCCR project files
include(projectdir("src", "setup.jl"))

# Point to the save directory
savedir(args...) = results_dir("processed", args...)

# Initialize container for the dataframes
dfs = Dict{String, DataFrame}()
# Iterate over every file name
for metric_file in readdir(savedir())
    # Point to the full file name
    savefile = savedir(metric_file)
    # Get the name of the metric by removing the extension
    metric, _ = splitext(metric_file)
    # Load the df and add to the dictionary of dfs
    dfs[metric] = DataFrame(CSV.File(savefile))
end

# Names for the rows of the latex table
pretty_rows = Dict(
    "art_activation" => "Activation",
    "art_match" => "Match",
    "performance" => "Performance",
)

# Initialize the output statistics dataframe
out_df = DataFrame(
    Metric=String[],
    PM=String[],
    FTR=String[],
    BTR=String[]
)

# Point to each l2metric symbol that we want to use here
syms = [:pm, :ftr, :btr]
# Iterate over every metric
for (metric, df) in dfs
    # Create an empty new entry for the stats df
    new_entry = String[]
    # First entry is the pretty name of the metric
    push!(new_entry, pretty_rows[metric])
    # Iterate over every l2metric symbol
    for sym in syms
        push!(new_entry, "$(mean(df[:, sym])) ± $(var(df[:, sym]))")
    end
    # for ix = 1:length(syms)
    #     push!(new_entry, "$(mean(df[:, syms[ix]])) ± $(var(df[:, syms[ix]]))")
    # end
    # Add the entry to the output stats df
    push!(out_df, new_entry)
end

# Make a latex version of the stats dataframe and save
new_df_tex = latexify(out_df, env=:table, fmt="%.3f")
table_file = paper_results_dir("perm_stats.tex")
open(table_file, "w") do f
    write(f, new_df_tex)
end