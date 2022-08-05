"""
    7_analyze.jl

Description:
    This script takes the results of the Monte Carlo of shuffled simulations
and generates plots of their statistics.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 6/24/2022: Created and documented.
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

using Revise
using DrWatson

# Experiment save directory name
experiment_top = "7_unsupervised_mc"

# Run the common setup methods (data paths, etc.)
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Plot file names
n_w_plot_name = "7_n_w.png"
n_F2_plot_name = "7_n_F2.png"
perf_plot_name = "7_perf.png"
heatmap_plot_name = "7_heatmap.png"
diff_perf_plot_name = "7_diff_perf.png"

# Point to the local sweep data directory
sweep_dir = projectdir("work", "results", experiment_top, "sweep")

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)
# df = collect_results(sweep_dir)

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
p_perf = create_boxplot(perf_matrix, class_labels, percentages=true, bounds_override=(0.8, 1))
ylabel!("Context Testing Accuracy")
display(p_perf)
# Save the plot
savefig(p_perf, results_dir(perf_plot_name))
savefig(p_perf, paper_results_dir(perf_plot_name))

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
norm_cm_df = df[:, :norm_cm]
norm_cm = mean(norm_cm_df)
h = create_custom_confusion_heatmap(class_labels, norm_cm)
display(h)

# Save the heatmap to both the local and paper results directories
savefig(h, results_dir(heatmap_plot_name))
savefig(h, paper_results_dir(heatmap_plot_name))


# -----------------------------------------------------------------------------
# DIFFERENCE PLOTS
# -----------------------------------------------------------------------------

# Training to testing difference performance
diff_perf_matrix = df_column_to_matrix(df, :a_te) .- df_column_to_matrix(df, :a_tev)[:, 1:6]
p_diff_perf = create_boxplot(diff_perf_matrix, class_labels, percentages=true, bounds_override=(-0.1, 0.1))
ylabel!("Context Testing Accuracy Difference")
display(p_diff_perf)
# Save the plot
savefig(p_diff_perf, results_dir(diff_perf_plot_name))
savefig(p_diff_perf, paper_results_dir(diff_perf_plot_name))

# # Normalized confusion heatmap
# # norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
# norm_cm_df_diff = df[:, :norm_cm]
# norm_cm = mean(norm_cm_df)
# h = create_custom_confusion_heatmap(class_labels, norm_cm)
# display(h)

# # Save the heatmap to both the local and paper results directories
# savefig(h, results_dir(heatmap_plot_name))
# savefig(h, paper_results_dir(heatmap_plot_name))