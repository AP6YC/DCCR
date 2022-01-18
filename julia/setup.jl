"""
    exp_setup.jl

Run common setup tasks for Julia experiments, such as setting loading source files and setting data paths.
"""

using Dates
using DrWatson
using Logging

# palette(:okabe_ito)

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Get the simulation datetime for the destination directory
sim_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Load source files
include(projectdir("julia", "lib_c3.jl"))

# Results directories (local and paper)
results_dir(args...) = projectdir("work", "results", experiment_top, args...)
paper_results_dir(args...) = joinpath("C:\\", "Users", "Sasha", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3", "images", "results", experiment_top, args...)

# Make the results directories if they do not exist
mkpath(results_dir())
mkpath(paper_results_dir())

# Top data directory
# data_dir = "E:\\dev\\mount\\data\\dist\\M18_Data_Drop_3_PR\\Data\\activations_yolov3"
# data_dir = joinpath("E:", "dev", "mount", "data", "dist", "M18_Data_Drop_3_PR", "Data", "activations_yolov3")
data_dir = joinpath("E:\\", "dev", "mount", "data", "dist", "M18_Data_Drop_3_PR", "Data", "activations_yolov3")
