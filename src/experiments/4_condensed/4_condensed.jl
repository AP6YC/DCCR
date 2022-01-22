"""
    4_condensed.jl

Description:
    This script runs a single condensed scenario iteration.

Author: Sasha Petrenko <sap625@mst.edu>
Date: 1/18/2022
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.

# Experiment save directory name
experiment_top = "4_condensed"

# Run the common setup methods (data paths, etc.)
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name = "4_condensed.png"
# plot_name_2 = "4_condensed_2.png"

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

# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

# Load the data names and class labels from the selection
data_dirs, class_labels = get_orbit_names(data_selection)

# Number of classes
n_classes = length(data_dirs)

# Load the data
data = load_orbits(data_dir, scaling)

# Sort/reload the data as indexed components
data_indexed = get_indexed_data(data)

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts)
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Get the data dimensions
dim, n_train = size(data.train_x)
_, n_test = size(data.test_x)

# Create the estimate containers
perfs = [[] for i = 1:n_classes]

# Initial testing block
for j = 1:n_classes
    push!(perfs[j], 0.0)
end

# Iterate over each class
for i = 1:n_classes
    _, n_samples_local = size(data_indexed.train_x[i])
    train!(ddvfa, data_indexed.train_x[i], y=data_indexed.train_y[i])

    # Test over each class
    for j = 1:n_classes
        local_y_hat = AdaptiveResonance.classify(ddvfa, data_indexed.test_x[j], get_bmu=true)
        push!(perfs[j], performance(local_y_hat, data_indexed.test_y[j]))
    end
end

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Simplified condensed scenario plot
p = create_condensed_plot(perfs, class_labels)
display(p)

# Save the plot
savefig(p, results_dir(plot_name))
savefig(p, paper_results_dir(plot_name))


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