"""
    exp_setup.jl

Run common setup tasks for Julia experiments, such as setting loading source files and setting data paths.
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
paper_results_dir(args...) = joinpath("C:\\", "Users", "Sasha", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3", "images", "results", experiment_top, args...)

# Make the results directories if they do not exist
mkpath(results_dir())
mkpath(paper_results_dir())

# Top data directory
# data_dir = joinpath("E:\\", "dev", "mount", "data", "dist", "M18_Data_Drop_3_PR", "Data", "activations_yolov3")
data_dir = projectdir("work", "data", "activations_yolov3")

# Plotting DPI
DPI = 350

# Plotting colorscheme
COLORSCHEME = :okabe_ito

# Heatmap color gradient
GRADIENTSCHEME = pubu_9[5:end]
# GRADIENTSCHEME = :thermal
# GRADIENTSCHEME = ylgn_9
# cgrad([:orange, :blue], [0.1, 0.3, 0.8])

# Plotting fontfamily for all text
FONTFAMILY = "Computer Modern"

# Aspect ratio correction for heatmap
# SQUARE_SIZE = 500.0 .* (1.0, 0.925)
SQUARE_SIZE = 500.0 .* (1.0, 0.94)

# Condensed plot parameters
# DOUBLE_WIDE = 0.75.* (1200, 400)
DOUBLE_WIDE = 1.0 .* (1200, 400)
# N_EB = 10
N_EB = 8

# colorbar_formatter
percentage_formatter = j -> @sprintf("%0.0f%%", 100*j)
