"""
    4_condensed_complex.jl

# Description
This script runs a complex single condensed scenario iteration.

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

using
    ProgressMeter,
    JLD2

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "4_condensed"

# Saving names
# plot_name = "4_condensed_complex.png"
data_file = DCCR.results_dir("condensed_complex_data.jld2")

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
    similarity = :single,
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
data = load_orbits(data_dir, data_dirs, scaling)

# Sort/reload the data as indexed components
data_indexed = get_indexed_data(data)

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts)
ddvfa.opts.display = false
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Get the data dimensions
dim, n_train = size(data.train.x)
_, n_test = size(data.test_x)

# Create the estimate containers
perfs = [[] for i = 1:n_classes]

# vals = [[] for i = 1:n_classes]

# Initial testing block
for j = 1:n_classes
    push!(perfs[j], 0.0)
end

vals = []
test_interval = 20

# Iterate over each class
for i = 1:n_classes
    # Learning block
    _, n_samples_local = size(data_indexed.train_x[i])
    # local_vals = zeros(n_classes, n_samples_local)
    # local_vals = zeros(n_classes, 0)
    local_vals = Array{Float64}(undef, n_classes, 0)

    # Iterate over all samples
    @showprogress for j = 1:n_samples_local
        train!(ddvfa, data_indexed.train_x[i][:, j], y=data_indexed.train_y[i][j])

        # Validation intervals
        if j % test_interval == 0
            # Validation data
            # local_y_hat = AdaptiveResonance.classify(ddvfa, data.val_x, get_bmu=true)
            # local_val = get_accuracies(data.val_y, local_y_hat, n_classes)
            # Training data
            local_y_hat = AdaptiveResonance.classify(ddvfa, data.train.x, get_bmu=true)
            local_val = get_accuracies(data.train.y, local_y_hat, n_classes)
            local_vals = hcat(local_vals, local_val')
        end
    end

    push!(vals, local_vals)

    # Experience block
    for j = 1:n_classes
        local_y_hat = AdaptiveResonance.classify(ddvfa, data_indexed.test_x[j], get_bmu=true)
        push!(perfs[j], performance(local_y_hat, data_indexed.test_y[j]))
    end
end

# Clean the NaN vals
for i = 1:n_classes
    replace!(vals[i], NaN => 0.0)
end

# Save the data
jldsave(data_file; perfs, vals, class_labels)
