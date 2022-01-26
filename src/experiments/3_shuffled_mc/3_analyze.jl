"""
    3_analyze.jl

Description:
    This script takes the results of the Monte Carlo of shuffled simulations
and generates plots of their statistics.

Author: Sasha Petrenko <sap625@mst.edu>
Date: 1/17/2022
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

using Revise
using DrWatson

# Experiment save directory name
experiment_top = "3_shuffled_mc"

# Run the common setup methods (data paths, etc.)
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Plot file names
n_w_plot_name = "3_n_w.png"
n_F2_plot_name = "3_n_F2.png"
perf_plot_name = "3_perf.png"
heatmap_plot_name = "3_heatmap.png"

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
p_F2 = create_boxplot(n_F2_matrix, class_labels)
ylabel!("# F2 Nodes")
display(p_F2)
# Save the plot
savefig(p_F2, results_dir(n_F2_plot_name))
savefig(p_F2, paper_results_dir(n_F2_plot_name))

# Testing performance
perf_matrix = df_column_to_matrix(df, :a_te)
p_perf = create_boxplot(perf_matrix, class_labels, percentages=true)
ylabel!("Class Testing Accuracy")
display(p_perf)
# Save the plot
savefig(p_perf, results_dir(perf_plot_name))
savefig(p_perf, paper_results_dir(perf_plot_name))

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
norm_cm_df = df[:, :norm_cm]
norm_cm = mean(cms)
h = create_custom_confusion_heatmap(class_labels, norm_cm)
display(h)

# Save the heatmap to both the local and paper results directories
savefig(h, results_dir(heatmap_plot_name))
savefig(h, paper_results_dir(heatmap_plot_name))
