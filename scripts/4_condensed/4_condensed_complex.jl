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
data_file = DCCR.results_dir(experiment_top, "condensed_complex_data.jld2")

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

# Sort/reload the data as indexed components
data_indexed = DCCR.get_indexed_data(data)

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts["opts_DDVFA"])
ddvfa.opts.display = false
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Get the data dimensions
dim, n_train = size(data.train.x)
_, n_test = size(data.test.x)

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
    _, n_samples_local = size(data_indexed.train.x[i])
    # local_vals = zeros(n_classes, n_samples_local)
    # local_vals = zeros(n_classes, 0)
    local_vals = Array{Float64}(undef, n_classes, 0)

    # Iterate over all samples
    @showprogress for j = 1:n_samples_local
        train!(ddvfa, data_indexed.train.x[i][:, j], y=data_indexed.train.y[i][j])

        # Validation intervals
        if j % test_interval == 0
            # Validation data
            # local_y_hat = AdaptiveResonance.classify(ddvfa, data.val_x, get_bmu=true)
            # local_val = get_accuracies(data.val_y, local_y_hat, n_classes)
            # Training data
            local_y_hat = AdaptiveResonance.classify(ddvfa, data.train.x, get_bmu=true)
            local_val = DCCR.get_accuracies(data.train.y, local_y_hat, n_classes)
            local_vals = hcat(local_vals, local_val')
        end
    end

    push!(vals, local_vals)

    # Experience block
    for j = 1:n_classes
        local_y_hat = AdaptiveResonance.classify(ddvfa, data_indexed.test.x[j], get_bmu=true)
        push!(perfs[j], performance(local_y_hat, data_indexed.test.y[j]))
    end
end

# Clean the NaN vals
for i = 1:n_classes
    replace!(vals[i], NaN => 0.0)
end

# Save the data
# DCCR.save_sim_results(data_file, perfs, vals, class_labels)
jldsave(data_file; perfs, vals, class_labels)
