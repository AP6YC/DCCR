"""
    4_analyze_complex.jl

# Description
This script analyzes a complex single condensed scenario iteration.

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
plot_name = "4_condensed_complex.png"

# Load name
data_file = DCCR.results_dir(experiment_top, "condensed_complex_data.jld2")

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

# Parse the arguments provided to this script
pargs = DCCR.exp_parse(
    "4_analyze_complex: full condensed scenario plot."
)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

# Load the data used for generating the condensed scenario plot
perfs, vals, class_labels = DCCR.load_sim_results(data_file, "perfs", "vals", "class_labels")

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Simplified condensed scenario plot
# p = create_condensed_plot(perfs, class_labels)
p, plot_data = DCCR.create_complex_condensed_plot(perfs, vals, class_labels)
DCCR.handle_display(p, pargs)

# Save the plot
DCCR.save_dccr("figure", p, experiment_top, plot_name, to_paper=pargs["paper"])
