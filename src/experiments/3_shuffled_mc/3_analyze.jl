"""
    3_analyze.jl

Description:
    This script takes the results of the Monte Carlo of shuffled simulations
and generates plots of their statistics.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 1/17/2022: Created.
- 2/17/2022: Documented.
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DrWatson
@quickactivate :DCCR

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "3_shuffled_mc"

# Flag for saving to paper results directory
SAVE_TO_PAPER_DIR = false

# Flag for displaying plots
DISPLAY = false

# Simulation options
opts_file = "default.yml"

# Plot file names
n_w_plot_name = "3_n_w.png"
n_F2_plot_name = "3_n_F2.png"
perf_plot_name = "3_perf.png"
heatmap_plot_name = "3_heatmap.png"

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Load the default simulation options
opts = DCCR.load_sim_opts(opts_file)

# Unpack the data if it is isn't already
DCCR.safe_unpack(experiment_top)

# Point to the local sweep data directory
sweep_dir = DCCR.unpacked_dir(experiment_top, "sweep")

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)

# Load the data names and class labels from the selection
data_dirs, class_labels = DCCR.get_orbit_names(opts["data_selection"])

# df_high = filter(row -> row[:test_perf] > 0.97, df)
# df_high[:, Not([:method, :arch, :path])]

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Total number of weights
n_w_matrix = DCCR.df_column_to_matrix(df, :n_w)
p_w = DCCR.create_boxplot(n_w_matrix, class_labels)
ylabel!("# Weights")
# Display if the option is set
DISPLAY && display(p_w)
# Save the plot
DCCR.save_dccr("figure", p_w, experiment_top, n_w_plot_name, to_paper=SAVE_TO_PAPER_DIR)


# Number of F2 nodes
n_F2_matrix = DCCR.df_column_to_matrix(df, :n_F2)
p_F2 = DCCR.create_boxplot(n_F2_matrix, class_labels)
ylabel!("# F2 Nodes")
# Display if the option is set
DISPLAY && display(p_F2)
# Save the plot
DCCR.save_dccr("figure", p_F2, experiment_top, n_F2_plot_name, to_paper=SAVE_TO_PAPER_DIR)

# Testing performance
perf_matrix = DCCR.df_column_to_matrix(df, :a_te)
p_perf = DCCR.create_boxplot(perf_matrix, class_labels, percentages=true)
ylabel!("Context Testing Accuracy")
# Display if the option is set
DISPLAY && display(p_perf)
# Save the plot
DCCR.save_dccr("figure", p_perf, experiment_top, perf_plot_name, to_paper=SAVE_TO_PAPER_DIR)

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
norm_cm_df = df[:, :norm_cm]
norm_cm = mean(norm_cm_df)
h = DCCR.create_custom_confusion_heatmap(class_labels, norm_cm)
# Display if the option is set
DISPLAY && display(h)
# Save the heatmap to both the local and paper results directories
DCCR.save_dccr("figure", h, experiment_top, heatmap_plot_name, to_paper=SAVE_TO_PAPER_DIR)
