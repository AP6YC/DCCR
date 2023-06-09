"""
    2_shuffled.jl

# Description
This script does the same experiment as 1_unshuffled.jl but with shuffling
the individual samples. The same training, testing, plots, and category tables
are generated and saved.

# Authors
- Sasha Petrenko <sap625@mst.edu>
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
experiment_top = "2_shuffled"

# Flag for saving to paper results directory
SAVE_TO_PAPER_DIR = false

# Flag for displaying plots
DISPLAY = false

# Simulation options
opts_file = "default.yml"

# Saving names
plot_name = "2_accuracy_shuffled.png"
heatmap_name = "2_heatmap.png"
n_cat_name = "2_n_cat.tex"

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the default simulation options
opts = DCCR.load_sim_opts(opts_file)

# Load the data names and class labels from the selection
data_dirs, class_labels = DCCR.get_orbit_names(opts["data_selection"])

# Number of classes
n_classes = length(data_dirs)

# Load the orbit data
data = DCCR.load_orbits(DCCR.data_dir, data_dirs, opts["scaling"])

i_train = randperm(length(data.train.y))
local_x = data.train.x[:, i_train]
local_y = data.train.y[i_train]

# (X_train, y_train), (X_test, y_test) = stratifiedobs((data, targets))

ddvfa = DDVFA(opts["opts_DDVFA"])
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Train in batch
y_hat_train = train!(ddvfa, local_x, y=local_y)
y_hat = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

# Calculate performance on training data, testing data, and with get_bmu
perf_train = performance(y_hat_train, local_y)
perf_test = performance(y_hat, data.test.y)

# Format each performance number for comparison
@printf "Batch training performance: %.4f\n" perf_train
@printf "Batch testing performance: %.4f\n" perf_test

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Create an accuracy grouped bar chart
p = DCCR.create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels, percentages=true)
# Display if the option is set
DISPLAY && display(p)
# Save the plot
DCCR.save_dccr("figure", p, experiment_top, plot_name, to_paper=SAVE_TO_PAPER_DIR)

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
DCCR.save_dccr("table", table, experiment_top, n_cat_name, to_paper=SAVE_TO_PAPER_DIR)

# -----------------------------------------------------------------------------
# CONFUSION HEATMAP
# -----------------------------------------------------------------------------

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
h = DCCR.create_confusion_heatmap(class_labels, data.test.y, y_hat)
# Display if the option is set
DISPLAY && display(h)
# Save the heatmap to both the local and paper results directories
DCCR.save_dccr("figure", h, experiment_top, heatmap_name, to_paper=SAVE_TO_PAPER_DIR)
