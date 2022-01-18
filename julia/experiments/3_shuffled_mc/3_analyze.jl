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

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "3_shuffled_mc"

# Run the common setup methods (data paths, etc.)
include(projectdir("julia", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Plot file names
n_w_plot_name = "3_n_w.png"
n_F2_plot_name = "3_n_F2.png"
perf_plot_name = "3_perf.png"

# Point to the local sweep data directory
sweep_dir = projectdir("work", "results", "3_shuffled_mc", "sweep")

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Total number of weights
n_w_matrix = df_column_to_matrix(df, :n_w)
p_w = create_boxplot(n_w_matrix, class_labels)
ylabel!("# Weights")
display(p_w)
# Save the plot
savefig(p_w, results_dir(n_w_plot_name))
savefig(p_w, paper_results_dir(n_w_plot_name))

# Number of F2 nodes
n_F2_matrix = df_column_to_matrix(df, :n_F2)
ylabel!("# F2 Nodes")
p_F2 = create_boxplot(n_F2_matrix, class_labels)
display(p_F2)
# Save the plot
savefig(p_F2, results_dir(n_F2_plot_name))
savefig(p_F2, paper_results_dir(n_F2_plot_name))

# Testing performance
perf_matrix = df_column_to_matrix(df, :a_te)
ylabel!("Performance")
p_perf = create_boxplot(perf_matrix, class_labels)
display(p_perf)
# Save the plot
savefig(p_perf, results_dir(perf_plot_name))
savefig(p_perf, paper_results_dir(perf_plot_name))
