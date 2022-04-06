"""
    2_shuffled.jl

Description:
    This script does the same experiment as 1_unshuffled.jl but with shuffling
the individual samples. The same training, testing, plots, and category tables
are generated and saved.

Timeline:
- 1/17/2022: Created.
- 2/17/2022: Documented.
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

# Load Revise for speed and DrWatson for folder pointing
using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.

# Experiment save directory name
experiment_top = "2_shuffled"

# Run the common setup methods (data paths, etc.)
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name = "2_accuracy_shuffled.png"
heatmap_name = "2_heatmap.png"

# n_F2_name = "1_n_F2.tex"
n_cat_name = "2_n_cat.tex"

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

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the data names and class labels from the selection
data_dirs, class_labels = get_orbit_names(data_selection)

# Number of classes
n_classes = length(data_dirs)

# X_train, y_train, train_labels, X_test, y_test, test_labels = load_orbits(data_dir, scaling)
data = load_orbits(data_dir, scaling)

i_train = randperm(length(data.train_y))
data.train_x = data.train_x[:, i_train]
data.train_y = data.train_y[i_train]

# (X_train, y_train), (X_test, y_test) = stratifiedobs((data, targets))

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
# PLOTTING
# -----------------------------------------------------------------------------

# Create an accuracy grouped bar chart
p = create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels, percentages=true)
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
