"""
    1_unshuffled.jl

Description:
    This file runs a single condensed scenario without experience blocks,
getting the final accuracies per category and saving them to a combined bar chart.
This script also counts the number of categories, saving to a LaTeX table.
Both of these results are saved to a local results directory and Dropbox directory
containing the Overleaf document for the paper.

Author: Sasha Petrenko <sap625@mst.edu>
Date: 1/17/2022
"""

using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Logging           # Printing diagnostics
using AdaptiveResonance # ART modules
using Random            # Random subsequence
# using ProgressMeter     # Progress bar
# using CSV
# using DataFrames
# using Dates
using MLDataUtils
using Printf            # Formatted number printing
# using JSON
# using MLBase
# using Plots
# using StatsPlots

using Latexify
using DataFrames

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "1_accuracy"

# Run the common setup methods (data paths, etc.)
include(projectdir("julia", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name = "1_accuracy.png"
heatmap_name = "1_heatmap.png"

# n_F2_name = "1_n_F2.tex"
n_cat_name = "1_n_cat.tex"

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

# Create the DDVFA options
opts = opts_DDVFA(
    gamma = 5.0,
    gamma_ref = 1.0,
    rho_lb = 0.45,
    rho_ub = 0.7,
    method = "single"
)

# Sigmoid input scaling
scaling = 2.0

# Plotting DPI
dpi = 350

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the data names and class labels from the selection
data_dirs, class_labels = get_orbit_names(data_selection)

# Number of classes
n_classes = length(data_dirs)

# Load the data
data = load_orbits(data_dir, scaling)

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts)
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Train in batch
y_hat_train = train!(ddvfa, data.train_x, y=data.train_y)
y_hat = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

# Calculate performance on training data, testing data, and with get_bmu
perf_train = performance(y_hat_train, data.train_y)
perf_test = performance(y_hat, data.test_y)

# Format each performance number for comparison
@printf "Batch training performance: %.4f\n" perf_train
@printf "Batch testing performance: %.4f\n" perf_test

# -----------------------------------------------------------------------------
# ACCURACY PLOTTING
# -----------------------------------------------------------------------------

# Create an accuracy grouped bar chart
p = create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels)
display(p)

# Save the plot
savefig(p, results_dir(plot_name))
savefig(p, paper_results_dir(plot_name))

# -----------------------------------------------------------------------------
# CATEGORY ANALYSIS
# -----------------------------------------------------------------------------

# Save the number of F2 nodes and total categories per class
n_F2, n_categories = get_n_categories(ddvfa)
@info "F2 nodes:" n_F2
@info "Total categories:" n_categories

# Create a LaTeX table from the categories
df = DataFrame(F2 = n_F2, Total = n_categories)
table = latexify(df, env=:table)

# Save the categories table to both the local and paper results directories
open(results_dir(n_cat_name), "w") do io
    write(io, table)
end
open(paper_results_dir(n_cat_name), "w") do io
    write(io, table)
end

# -----------------------------------------------------------------------------
# CONFUSION HEATMAP
# -----------------------------------------------------------------------------

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
h = create_confusion_heatmap(class_labels, data.test_y, y_hat)
display(h)

# Save the heatmap to both the local and paper results directories
savefig(h, results_dir(heatmap_name))
savefig(h, paper_results_dir(heatmap_name))
