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

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Get the simulation datetime for the destination directory
sim_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Load source files
include(projectdir("julia", "lib_c3.jl"))

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Create the DDVFA options
opts = opts_DDVFA(
    gamma = 5.0,
    gamma_ref = 1.0,
    # rho=0.45,
    rho_lb = 0.45,
    rho_ub = 0.7,
    method = "single"
)

# Sigmoid input scaling
scaling = 2.0

# Data directories to train/test on
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

data_dir = "E:\\dev\\mount\\data\\dist\\M18_Data_Drop_3_PR\\Data\\activations_yolov3"

train_dir = joinpath(data_dir, "LBs")
test_dir = joinpath(data_dir, "EBs")


train_data_dirs = [joinpath(train_dir, data_dir) for data_dir in data_dirs]
test_data_dirs = [joinpath(test_dir, data_dir) for data_dir in data_dirs]

# joinpath(data_dir_src, data_dirs)
X_train, y_train, train_labels = collect_all_activations_labeled(train_data_dirs, 1)
X_test, y_test, test_labels = collect_all_activations_labeled(test_data_dirs, 1)

dt = get_dist(X_train)

X_train = feature_preprocess(dt, scaling, X_train)
X_test = feature_preprocess(dt, scaling, X_test)


# i_train = randperm(length(y_train))
# X_train = X_train[:, i_train]
# y_train = y_train[i_train]

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
