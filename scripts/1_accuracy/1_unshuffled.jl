"""
    1_unshuffled.jl

# Description
This file runs a single condensed scenario without experience blocks,
getting the final accuracies per category and saving them to a combined bar chart.
This script also counts the number of categories, saving to a LaTeX table.
Both of these results are saved to a local results directory and Dropbox directory
containing the Overleaf document for the paper.

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

using AdaptiveResonance
using DataFrames
using Printf

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "1_accuracy"

# Simulation options
opts_file = "default.yml"

# Saving names
plot_name = "1_accuracy.png"
heatmap_name = "1_heatmap.png"
n_cat_name = "1_n_cat.tex"

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

# Parse the arguments provided to this script
pargs = DCCR.exp_parse(
    "1_accuracy: unshuffled condensed scenario"
)

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the default simulation options
opts = DCCR.load_sim_opts(opts_file)

# Load the data names and class labels from the selection
data_dirs, class_labels = DCCR.get_orbit_names(opts["data_selection"])

# Number of classes
n_classes = length(data_dirs)

# Load the data
data = DCCR.load_orbits(DCCR.data_dir, data_dirs, opts["scaling"])

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts["opts_DDVFA"])
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Train in batch
y_hat_train = train!(ddvfa, data.train.x, y=data.train.y)
y_hat = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

# Calculate performance on training data, testing data, and with get_bmu
perf_train = performance(y_hat_train, data.train.y)
perf_test = performance(y_hat, data.test.y)

# Format each performance number for comparison
@printf "Batch training performance: %.4f\n" perf_train
@printf "Batch testing performance: %.4f\n" perf_test

# -----------------------------------------------------------------------------
# ACCURACY PLOTTING
# -----------------------------------------------------------------------------

# Create an accuracy grouped bar chart
p = DCCR.create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels, percentages=true)
DCCR.handle_display(p, pargs)

# Save the plot
DCCR.save_dccr("figure", p, experiment_top, plot_name, to_paper=pargs["paper"])

# -----------------------------------------------------------------------------
# CATEGORY ANALYSIS
# -----------------------------------------------------------------------------

# Save the number of F2 nodes and total categories per class
n_F2, n_categories = DCCR.get_n_categories(ddvfa, n_classes)
@info "F2 nodes:" n_F2
@info "Total categories:" n_categories

# Create a LaTeX table from the categories
df = DataFrame(F2 = n_F2, Total = n_categories)
table = latexify(df, env=:table)

# Save the categories table to both the local and paper results directories
DCCR.save_dccr("table", table, experiment_top, n_cat_name, to_paper=pargs["paper"])

# -----------------------------------------------------------------------------
# CONFUSION HEATMAP
# -----------------------------------------------------------------------------

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
h = DCCR.create_confusion_heatmap(class_labels, data.test.y, y_hat)
DCCR.handle_display(h, pargs)

# Save the heatmap to both the local and paper results directories
DCCR.save_dccr("figure", h, experiment_top, heatmap_name, to_paper=pargs["paper"])
