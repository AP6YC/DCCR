"""
    exp_setup.jl

Run common setup tasks for Julia experiments, such as setting loading source files and setting data paths.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 1/15/2022: Created.
- 2/17/2022: Documented.
"""

using DrWatson
using Dates
using Logging           # Printing diagnostics
using Random            # Random subsequence
using Plots

using AdaptiveResonance # ART modules
# using ProgressMeter     # Progress bar
using Printf            # Formatted number printing

using Latexify          #
using DataFrames

# using PlotThemes
# Plotting style
# pyplot()
# theme(:dark)

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Get the simulation datetime for the destination directory
sim_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Load source files
include(projectdir("src", "lib_c3.jl"))

# Results directories (local and paper)
results_dir(args...) = projectdir("work", "results", experiment_top, args...)
paper_results_dir(args...) = joinpath("C:\\", "Users", "Sasha", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3-V3-Resubmission", "images", "results", experiment_top, args...)
# paper_results_dir(args...) = joinpath("C:\\", "Users", "sap62", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3-V3-Resubmission", "images", "results", experiment_top, args...)

# Make the results directories if they do not exist
mkpath(results_dir())
mkpath(paper_results_dir())

# Top data directory
const data_dir = projectdir("work", "data", "activations_yolov3_cell=1")

# Plotting DPI
const DPI = 350

# Plotting colorscheme
const COLORSCHEME = :okabe_ito

# Heatmap color gradient
if !isdefined(Main, :GRADIENTSCHEME)
    const GRADIENTSCHEME = pubu_9[5:end]
end
# GRADIENTSCHEME = :thermal
# GRADIENTSCHEME = ylgn_9
# cgrad([:orange, :blue], [0.1, 0.3, 0.8])

# Plotting fontfamily for all text
const FONTFAMILY = "Computer Modern"

# Aspect ratio correction for heatmap
# SQUARE_SIZE = 500.0 .* (1.0, 0.925)
# SQUARE_SIZE = 500.0 .* (1.0, 0.94)
# SQUARE_SIZE = 500.0 .* (1.0, 0.86)  # -9Plots.mm
const SQUARE_SIZE = 500.0 .* (1.0, 0.87)  # -8Plots.mm

# Condensed plot parameters
# DOUBLE_WIDE = 0.75.* (1200, 400)
const DOUBLE_WIDE = 1.0 .* (1200, 400)
# N_EB = 10
const N_EB = 8
const CONDENSED_LINEWIDTH = 2.5

# colorbar_formatter
if !isdefined(Main, :percentage_formatter)
    const percentage_formatter = j -> @sprintf("%0.0f%%", 100*j)
end

const PERCENTAGES_BOUNDS = (0.45, 1)
