"""
    4_analyze_complex_alt.jl

Description:
    This script analyzes a complex single condensed scenario iteration.
This script is updated to use the updated full condensed scenario plot.

Timeline:
- 1/26/2022: Created.
- 2/17/2022: Documented.
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.

using JLD2

# Experiment save directory name
experiment_top = "4_condensed"

# Run the common setup methods (data paths, etc.)
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Saving names
plot_name = "4_condensed_complex_alt.png"

# Load name
data_file = results_dir("condensed_complex_data.jld2")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

perfs, vals, class_labels = load(data_file, "perfs", "vals", "class_labels")

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# p = create_condensed_plot(perfs, class_labels)
p, training_vals, x_training_vals = create_complex_condensed_plot_alt(perfs, vals, class_labels)
display(p)

# Save the plot
savefig(p, results_dir(plot_name))
savefig(p, paper_results_dir(plot_name))
