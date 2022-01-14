using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Logging           # Printing diagnostics
using AdaptiveResonance # ART modules
using Random            # Random subsequence
# using ProgressMeter     # Progress bar
# using CSV
# using DataFrames
using Dates
using MLDataUtils
using Printf            # Formatted number printing
# using JSON
using MLBase
# using Plots
using StatsPlots

using Latexify
using DataFrames

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "2_shuffled"

# Run the common setup methods (data paths, etc.)
include(projectdir("julia", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name = "2_accuracy_shuffled.png"
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

# Plotting DPI
dpi = 350

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

# TRAIN: Get the percent correct for each class
cm = confusmat(n_classes, data.train_y, y_hat_train)
correct = [cm[i,i] for i = 1:n_classes]
total = sum(cm, dims=1)
train_accuracies = correct'./total

# TEST: Get the percent correct for each class
cm = confusmat(n_classes, data.test_y, y_hat)
correct = [cm[i,i] for i = 1:n_classes]
total = sum(cm, dims=1)
test_accuracies = correct'./total

@info "Train Accuracies:" train_accuracies
@info "Train Accuracies:" test_accuracies

# Format the accuracy series for plotting
combined_accuracies = [train_accuracies; test_accuracies]'

# groupedbar(rand(10,3), bar_position = :dodge, bar_width=0.7)
p = groupedbar(
    combined_accuracies,
    bar_position = :dodge,
    bar_width=0.7,
    dpi=dpi,
    # show=true,
    # xticks=train_labels
)

ylabel!(p, "Accuracy")
xticks!(collect(1:n_classes), class_labels)
# title!(p, "test")

# Save the plot
savefig(p, results_dir(plot_name))
savefig(p, paper_results_dir(plot_name))

# -----------------------------------------------------------------------------
# CATEGORY ANALYSIS
# -----------------------------------------------------------------------------

# Save the number of F2 nodes and total categories per class
n_F2 = Int[]
n_categories = Int[]
# Iterate over every class
for i = 1:n_classes
    # Find all of the F2 nodes that correspond to the class
    i_F2 = findall(x->x==i, ddvfa.labels)
    # Add the number of F2 nodes to the list
    push!(n_F2, length(i_F2))
    # Get the numbers of categories within each F2 node
    n_cat_list = [F2.n_categories for F2 in ddvfa.F2[i_F2]]
    # Sum those and add them to the list
    push!(n_categories, sum(n_cat_list))
end

@info "F2 nodes:" n_F2
@info "Total categories:" n_categories

df = DataFrame(F2 = n_F2, Total = n_categories)
table = latexify(df, env=:table)

open(results_dir(n_cat_name), "w") do io
    write(io, table)
end

open(paper_results_dir(n_cat_name), "w") do io
    write(io, table)
end
