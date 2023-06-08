"""
    1_unshuffled.jl

Description:
    This file runs a single condensed scenario without experience blocks,
getting the final accuracies per category and saving them to a combined bar chart.
This script also counts the number of categories, saving to a LaTeX table.
Both of these results are saved to a local results directory and Dropbox directory
containing the Overleaf document for the paper.

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
# FILE SETUP
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "1_accuracy"

# Run the common setup methods (data paths, etc.)
# include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

SAVE_TO_PAPER_DIR = false
DISPLAY = false

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
    similarity = :single,
)

# Sigmoid input scaling
scaling = 2.0

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the data names and class labels from the selection
data_dirs, class_labels = DCCR.get_orbit_names(data_selection)

# Number of classes
n_classes = length(data_dirs)

# Load the data
data = DCCR.load_orbits(DCCR.data_dir, data_dirs, scaling)

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts)
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
DISPLAY && display(p)

# Save the plot
# savefig(p, results_dir(plot_name))
# SAVE_TO_PAPER_DIR && savefig(p, paper_results_dir(plot_name))
DCCR.save_dccr("figure", p, experiment_top, plot_name)

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

# DCCR.save_dccr("table", table, )

# # Save the categories table to both the local and paper results directories
# open(results_dir(n_cat_name), "w") do io
#     write(io, table)
# end

# if SAVE_TO_PAPER_DIR
#     open(paper_results_dir(n_cat_name), "w") do io
#         write(io, table)
#     end
# end
# # -----------------------------------------------------------------------------
# # CONFUSION HEATMAP
# # -----------------------------------------------------------------------------

# # Normalized confusion heatmap
# # norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
# h = create_confusion_heatmap(class_labels, data.test_y, y_hat)
# display(h)

# # Save the heatmap to both the local and paper results directories
# savefig(h, results_dir(heatmap_name))
# SAVE_TO_PAPER_DIR && savefig(h, paper_results_dir(heatmap_name))
