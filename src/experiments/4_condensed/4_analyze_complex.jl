"""
    4_analyze_complex.jl

Description:
    This script analyzes a complex single condensed scenario iteration.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 1/18/2022: Created.
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
plot_name = "4_condensed_complex.png"

# Load name
data_file = results_dir("condensed_complex_data.jld2")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

perfs, vals, class_labels = load(data_file, "perfs", "vals", "class_labels")

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Simplified condensed scenario plot
# p = create_condensed_plot(perfs, class_labels)
p, plot_data = create_complex_condensed_plot(perfs, vals, class_labels)
display(p)

# Save the plot
savefig(p, results_dir(plot_name))
savefig(p, paper_results_dir(plot_name))
