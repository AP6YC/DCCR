"""
    3_analyze.jl

Description:
    This script takes the results of the Monte Carlo of shuffled simulations
and generates plots of their statistics.

Author: Sasha Petrenko <sap625@mst.edu>
Date: 1/17/2022
"""

using Revise
using DataFrames
# using Plots
# using PlotThemes
using DrWatson
using StatsPlots
# Plotting style
# pyplot()
# theme(:dark)


# Experiment save directory name
experiment_top = "3_shuffled_mc"

n_w_plot_name = "3_n_w.png"
n_F2_plot_name = "3_n_F2.png"

# Run the common setup methods (data paths, etc.)
include(projectdir("julia", "setup.jl"))

# Point to the local sweep data directory
sweep_dir = projectdir("work", "results", "3_shuffled_mc", "sweep")

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)

# Select which data entries to use for the experiment
data_selection = [
    "dot_dusk",
    "dot_morning",
    # "emahigh_dusk",
    # "emahigh_morning",
    "emalow_dusk",
    "emalow_morning",
    "pr_dusk",
    "pr_morning",
]

# Load the data names and class labels from the selection
data_dirs, class_labels = get_orbit_names(data_selection)

# df_high = filter(row -> row[:test_perf] > 0.97, df)
# df_high[:, Not([:method, :arch, :path])]

# Analyse the number of weights per category
# n_w_lists = df[!, :n_w]
# n_samples = length(n_w_lists)
# n_classes = length(n_w_lists[1])
# n_w_matrix = zeros(n_samples, n_classes)
# for i = 1:n_samples
#     n_w_matrix[i, :] = n_w_lists[i]
# end

# Get a matrix from the column of number of weights
n_w_matrix = df_column_to_matrix(df, :n_w)
p_w = create_boxplot(n_w_matrix, class_labels)
ylabel!("# Weights")
display(p_w)

# Save the plot
savefig(p_w, results_dir(n_w_plot_name))
savefig(p_w, paper_results_dir(n_w_plot_name))


n_F2_matrix = df_column_to_matrix(df, :n_F2)
ylabel!("# F2 Nodes")
p_F2 = create_boxplot(n_F2_matrix, class_labels)
display(p_F2)

# Save the plot
savefig(p_F2, results_dir(n_F2_plot_name))
savefig(p_F2, paper_results_dir(n_F2_plot_name))


# df_high = filter(row -> row[:test_perf] > 0.94 && row[:train_perf] > 0.94, df)

# plot
# xlabel!("F2 Category")
# ylabel!("# Weights")
# title!(string(cell)*(cell > 1 ? " Windows" : " Window"))
# # fig_path = joinpath(cell, string(cell))
# fig_dir = joinpath("work/results", subdir)
# mkpath(fig_dir)
# fig_path = joinpath(fig_dir, string(cell))
# savefig(fig_path)
