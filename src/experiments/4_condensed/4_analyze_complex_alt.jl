"""
    4_analyze_complex_alt.jl

# Description
This script analyzes a complex single condensed scenario iteration.
This script is updated to use the updated full condensed scenario plot.

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
# ADDITIONAL DEPENDENCIES
# -----------------------------------------------------------------------------

using JLD2

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "4_condensed"

# Saving names
plot_name = "4_condensed_complex_alt.png"

# Load name
data_file = DCCR.results_dir(experiment_top, "condensed_complex_data.jld2")

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

# Parse the arguments provided to this script
pargs = DCCR.exp_parse(
    "4_analyze_complex_alt: alternative full condensed scenario plot."
)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

# Load the data used for generating the condensed scenario plot
perfs, vals, class_labels = JLD2.load(data_file, "perfs", "vals", "class_labels")

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# p = create_condensed_plot(perfs, class_labels)
p, training_vals, x_training_vals = DCCR.create_complex_condensed_plot_alt(perfs, vals, class_labels)
pargs["display"] && display(p)

# Save the plot
DCCR.save_dccr("figure", p, experiment_top, plot_name, to_paper=pargs["paper"])
