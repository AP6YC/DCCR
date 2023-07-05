"""
    6_analyze.jl

# Description
Analyze/plot the permuted condensed scenario runs from 6_permuted.jl.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DCCR

# -----------------------------------------------------------------------------
# ADDITIONAL DEPENDENCIES
# -----------------------------------------------------------------------------

# using DataFrames
using DrWatson      # collect_results!
using StatsBase
using Plots

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "6_permuted"

# Plot file names
n_w_plot_name = "6_n_w.png"
n_F2_plot_name = "6_n_F2.png"
perf_plot_name = "6_perf.png"
heatmap_plot_name = "6_heatmap.png"

# Simulation options
opts_file = "default.yml"

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

# Parse the arguments provided to this script
pargs = DCCR.exp_parse(
    "6_analyze: analyze the permuted condensed scenarios."
)

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the default simulation options
opts = DCCR.load_sim_opts(opts_file)

# Point to the local sweep data directory
DCCR.safe_unpack(experiment_top)
sweep_dir = DCCR.unpacked_dir(experiment_top, "sweep")

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)
# df = collect_results(sweep_dir)

# Load the data names and class labels from the selection
data_dirs, class_labels = DCCR.get_orbit_names(opts["data_selection"])

# df_high = filter(row -> row[:test_perf] > 0.97, df)
# df_high[:, Not([:method, :arch, :path])]

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Total number of weights
n_w_matrix = DCCR.df_column_to_matrix(df, :n_w)
p_w = DCCR.create_inverted_boxplot(n_w_matrix, class_labels)
ylabel!(p_w, "# Weights")
DCCR.handle_display(p_w, pargs)

# Save the plot
DCCR.save_dccr("figure", p_w, experiment_top, n_w_plot_name, to_paper=pargs["paper"])

# Number of F2 nodes
n_F2_matrix = DCCR.df_column_to_matrix(df, :n_F2)
p_F2 = DCCR.create_inverted_boxplot(n_F2_matrix, class_labels)
ylabel!(p_F2, "# F2 Nodes")
DCCR.handle_display(p_F2, pargs)

# Save the plot
DCCR.save_dccr("figure", p_F2, experiment_top, n_F2_plot_name, to_paper=pargs["paper"])

# Testing performance
perf_matrix = DCCR.df_column_to_matrix(df, :a_te)
p_perf = DCCR.create_inverted_boxplot(perf_matrix, class_labels, percentages=true)
ylabel!(p_perf, "Context Testing Accuracy")
DCCR.handle_display(p_perf, pargs)

# Save the plot
DCCR.save_dccr("figure", p_perf, experiment_top, perf_plot_name, to_paper=pargs["paper"])

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
norm_cm_df = df[:, :norm_cm]
norm_cm = mean(norm_cm_df)
h = DCCR.create_custom_confusion_heatmap(class_labels, norm_cm)
DCCR.handle_display(h, pargs)

# Save the plot
DCCR.save_dccr("figure", h, experiment_top, heatmap_plot_name, to_paper=pargs["paper"])
