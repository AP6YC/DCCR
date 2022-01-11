using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Logging           # Printing diagnostics
using AdaptiveResonance # ART modules
using Random            # Random subsequence
using ProgressMeter     # Progress bar
# using CSV
# using DataFrames
using Dates
using MLDataUtils
using Printf            # Formatted number printing
# using JSON

# using
# -----------------------------------------
# SETUP
# -----------------------------------------

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Get the simulation datetime for the destination directory
sim_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Load source files
include(projectdir("julia", "lib_c3.jl"))

# -----------------------------------------
# OPTIONS
# -----------------------------------------

# Create the DDVFA options
opts = opts_DDVFA()
opts.gamma = 5.0
opts.gamma_ref = 1.0
opts.rho = 0.45
opts.rho_lb = 0.45
opts.rho_ub = 0.7
opts.method = "average"

scaling = 3

data_dir = "E:\\dev\\mount\\data\\dist\\M18_Data_Drop_3_PR\\Data\\activations_yolov3"
train_dir = joinpath(data_dir, "LBs")
test_dir = joinpath(data_dir, "EBs")

data_dirs = [
    "dot_dusk",
    "dot_morning",
    "emahigh_dusk",
    "emahigh_morning",
    "emalow_dusk",
    "emalow_morning",
    "pr_dusk",
    "pr_morning"
]

train_data_dirs = [joinpath(train_dir, data_dir) for data_dir in data_dirs]
test_data_dirs = [joinpath(test_dir, data_dir) for data_dir in data_dirs]

# joinpath(data_dir_src, data_dirs)
X_train, y_train, train_labels = collect_all_activations_labeled(train_data_dirs, 1)
X_test, y_test, test_labels = collect_all_activations_labeled(test_data_dirs, 1)

# data_split = DataSplit(data, targets, 0.8)

dt = get_dist(X_train)

X_train = feature_preprocess(dt, scaling, X_train)
X_test = feature_preprocess(dt, scaling, X_test)

# (X_train, y_train), (X_test, y_test) = stratifiedobs((data, targets))

ddvfa = DDVFA(opts)
ddvfa.config = DataConfig(0, 1, 128)

# We can train in batch with a simple supervised mode by passing the labels as a keyword argument.
y_hat_train = train!(ddvfa, X_train, y=y_train)
# println("Training labels: ",  size(y_hat_batch_train), " ", typeof(y_hat_batch_train))
y_hat = AdaptiveResonance.classify(ddvfa, X_test, get_bmu=true)


## Calculate performance on training data, testing data, and with get_bmu
perf_train = performance(y_hat_train, y_train)
perf_test = performance(y_hat, y_test)

## Format each performance number for comparison
@printf "Batch training performance: %.4f\n" perf_train
@printf "Batch testing performance: %.4f\n" perf_test
