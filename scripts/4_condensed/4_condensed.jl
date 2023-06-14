"""
    4_condensed.jl

# Description
This script runs a single condensed scenario iteration.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DCCR

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "4_condensed"

# Saving names
plot_name = "4_condensed.png"
# plot_name_2 = "4_condensed_2.png"

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

# Parse the arguments provided to this script
pargs = DCCR.exp_parse(
    "4_condensed: a single condensed scenario iteration."
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

# Sort/reload the data as indexed components
data_indexed = DCCR.get_indexed_data(data)

# Create the DDVFA module and set the data config
ddvfa = DDVFA(opts["opts_DDVFA"])
ddvfa.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Get the data dimensions
dim, n_train = size(data.train.x)
_, n_test = size(data.test.x)

# Create the estimate containers
perfs = [[] for i = 1:n_classes]

# Initial testing block
for j = 1:n_classes
    push!(perfs[j], 0.0)
end

# Iterate over each class
for i = 1:n_classes
    # _, n_samples_local = size(data_indexed.train_x[i])
    train!(ddvfa, data_indexed.train.x[i], y=data_indexed.train.y[i])

    # Test over each class
    for j = 1:n_classes
        local_y_hat = AdaptiveResonance.classify(ddvfa, data_indexed.test.x[j], get_bmu=true)
        push!(perfs[j], performance(local_y_hat, data_indexed.test.y[j]))
    end
end

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Simplified condensed scenario plot
p = DCCR.create_condensed_plot(perfs, class_labels)
DCCR.handle_display(p, pargs)

# Save the plot
DCCR.save_dccr("figure", p, experiment_top, plot_name, to_paper=pargs["paper"])
