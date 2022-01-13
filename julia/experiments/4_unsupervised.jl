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

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

# Run the common setup methods (data paths, etc.)
include(projectdir("julia", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name_1 = "4_unsupervised_1.png"
plot_name_2 = "4_unsupervised_2.png"

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
    # rho=0.45,
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

# Number of classes
n_classes = length(data_dirs)

# Load the data names and class labels from the selection
data_dirs, class_labels = get_orbit_names(data_selection)

# Load the data
data = load_orbits(data_dir, scaling)

# Sort/reload the data as indexed components
data_indexed = get_indexed_data(data)


# # X_train, y_train, train_labels, X_test, y_test, test_labels = load_orbits(data_dir, scaling)
# data = load_orbits(data_dir, scaling)

# # i_train = randperm(length(y_train))
# # X_train = X_train[:, i_train]
# # y_train = y_train[i_train]

# # (X_train, y_train), (X_test, y_test) = stratifiedobs((data, targets))

ddvfa = DDVFA(opts)
ddvfa.config = DataConfig(0, 1, 128)

# # -----------------------------------------------------------------------------
# # TRAIN/TEST
# # -----------------------------------------------------------------------------

# # We can train in batch with a simple supervised mode by passing the labels as a keyword argument.
# y_hat_train = train!(ddvfa, data.train_x, y=data.train_y)
# # println("Training labels: ",  size(y_hat_batch_train), " ", typeof(y_hat_batch_train))
# y_hat = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

# # Calculate performance on training data, testing data, and with get_bmu
# perf_train = performance(y_hat_train, data.train_y)
# perf_test = performance(y_hat, data.test_y)

# # Format each performance number for comparison
# @printf "Batch training performance: %.4f\n" perf_train
# @printf "Batch testing performance: %.4f\n" perf_test

# # -----------------------------------------------------------------------------
# # PLOTTING
# # -----------------------------------------------------------------------------

# # TRAIN: Get the percent correct for each class
# cm = confusmat(n_classes, data.train_y, y_hat_train)
# correct = [cm[i,i] for i = 1:n_classes]
# total = sum(cm, dims=1)
# train_accuracies = correct'./total

# # TEST: Get the percent correct for each class
# cm = confusmat(n_classes, data.test_y, y_hat)
# correct = [cm[i,i] for i = 1:n_classes]
# total = sum(cm, dims=1)
# test_accuracies = correct'./total

# @info "Train Accuracies:" train_accuracies
# @info "Train Accuracies:" test_accuracies

# # Format the accuracy series for plotting
# combined_accuracies = [train_accuracies; test_accuracies]'

# # groupedbar(rand(10,3), bar_position = :dodge, bar_width=0.7)
# p = groupedbar(
#     combined_accuracies,
#     bar_position = :dodge,
#     bar_width=0.7,
#     dpi=dpi,
#     # show=true,
#     # xticks=train_labels
# )

# ylabel!(p, "Accuracy")
# xticks!(collect(1:n_classes), class_labels)
# # title!(p, "test")
# # Save the plot
# savefig(p, results_dir(plot_name))