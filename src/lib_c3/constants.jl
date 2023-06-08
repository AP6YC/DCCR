
using
    # DrWatson,
    Dates,
    Logging,           # Printing diagnostics
    Random,            # Random subsequence
    Plots,
    AdaptiveResonance, # ART modules
    Printf,            # Formatted number printing
    Latexify,
    DataFrames

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
# include(projectdir("src", "lib_c3.jl"))

# Top data directory
# const data_dir = projectdir("work", "data", "activations_yolov3_cell=1")
const data_dir = unpacked_dir("activations_yolov3_cell=1")

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
