"""
    5_unsupervised_val.jl

# Description
This script runs an unsupervised learning scenario. After supervised pretraining,
the module learns upon additional data unsupervised and is tested for performance
before and after.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DCCR
using DataFrames

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "5_unsupervised_val"

# Run the common setup methods (data paths, etc.)
# include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name = "5_accuracy.png"
heatmap_name = "5_heatmap.png"

# n_F2_name = "1_n_F2.tex"
n_cat_name = "5_n_cat.tex"

# Simulation options
opts_file = "default.yml"

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

# Parse the arguments provided to this script
pargs = DCCR.exp_parse(
    "5_unsupervised_val: supervised and unsupervised training and validation."
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
# data = load_orbits(data_dir, data_dirs, scaling)
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
@printf "Supervised training performance: %.4f\n" perf_train
@printf "Supervised testing performance: %.4f\n" perf_test

# -----------------------------------------------------------------------------
# UNSUPERVISED TRAIN/TEST ON VALIDATION DATA
# -----------------------------------------------------------------------------

# Train in batch, unsupervised
# y_hat_train_val = train!(ddvfa, data.val_x, y=data.val_y)
y_hat_train_val = train!(ddvfa, data.val.x)
# If the category is not in 1:6, replace the label as 7 for the new/incorrect bin
replace!(x -> !(x in collect(1:n_classes)) ? 7 : x, ddvfa.labels)
y_hat_val = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

# Calculate performance on training data, testing data, and with get_bmu
perf_train_val = performance(y_hat_train_val, data.val.y)
perf_test_val = performance(y_hat, data.test.y)

# Format each performance number for comparison
@printf "Unsupervised training performance: %.4f\n" perf_train_val
@printf "Unsupervised testing performance: %.4f\n" perf_test_val

# -----------------------------------------------------------------------------
# ACCURACY PLOTTING
# -----------------------------------------------------------------------------

# Create an accuracy grouped bar chart
p = DCCR.create_comparison_groupedbar(data, y_hat_val, y_hat, class_labels, percentages=true, extended=true)
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
# Display if the option is set
DCCR.handle_display(h, pargs)

# Save the heatmap to both the local and paper results directories
DCCR.save_dccr("figure", h, experiment_top, heatmap_name, to_paper=pargs["paper"])

