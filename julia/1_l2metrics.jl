using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Logging           # Printing diagnostics
using AdaptiveResonance # ART modules
using Random            # Random subsequence
using ProgressMeter     # Progress bar
using CSV
using DataFrames
using Dates
using JSON

# -----------------------------------------
# OPTIONS
# -----------------------------------------

# Extractor id
extractor_id = 3
# extractor_id = 1
# Using big data folder or compressed
# big_data = true
big_data = false

# Create the DDVFA options
opts = opts_DDVFA()
opts.gamma = 5.0
opts.gamma_ref = 1.0
opts.rho = 0.45
opts.rho_lb = 0.45
opts.rho_ub = 0.7
opts.method = "average"
# opts.rho = 0.7
# opts.rho_lb = 0.7
# opts.rho_ub = 0.85
# opts.method = "single"

# Scenario info
scenario_info = Dict(
    "complexity" => "2-intermediate",
    "scenario_type" => "condensed",
    "author" => "Sasha Petrenko",
    "difficulty" => "2-medium"
)

# Logger info
logger_info = Dict(
    "metrics_columns" => [
        "performance"
    ],
    "log_format_version"=> "1.0"
)

# -----------------------------------------
# SETUP
# -----------------------------------------

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Get the simulation datetime for the destination directory
sim_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Load source files
include(projectdir("experiments/lib_sim.jl"))

# Custom directories
flir_folder = big_data ? "flir_big_compressed" : "flir_compressed"
data_dir = projectdir("work", "results", flir_folder)
results_dir(args...) = projectdir("work", "results", "l2metrics", sim_datetime, args...)
scenario_dir(args...) = results_dir("0", args...)

# Define all extractors
extractors = readdir(joinpath(data_dir, "train"))
extractor = extractors[extractor_id]

# Make the scenario directory
mkpath(scenario_dir())

# Write the scenario info file
open(results_dir("scenario_info.json"), "w") do f
    # write(f, scenario_info)
    JSON.print(f, scenario_info)
end

# Write the logger info file
open(results_dir("logger_info.json"), "w") do f
    # write(f, logger_info)
    JSON.print(f, logger_info)
end

# -----------------------------------------
# SIM METHODS
# -----------------------------------------

# All of the logic for running one test or train loop, saving the results
function run_one(art::DDVFA, data::FLIRL2MSplitCombined, class_id::Integer, block_num::Integer, exp_num::Integer, file::String, mode::String)
    # labels = ["human", "bike", "car", "animal"]
    labels = ["human", "animal", "bike", "car"]

    if mode == "test"
        n_data = length(data.test_y[class_id])
        if !isempty(art.F2)
            y_hat_test = classify(art, data.test_x[class_id], get_bmu=true)
        else
            y_hat_test = repeat([-1], n_data)
        end

        # Create a performances destination that is explicitly a vector of floats
        performances = zeros(Float64, n_data)
        for i = 1:n_data
            performances[i] = (y_hat_test[i] == data.test_y[class_id][i])
        end
        # test_accuracies[ix] = AdaptiveResonance.performance(y_hat_test, data.train_y)
    elseif mode == "train"
        n_data = length(data.train_y[class_id])
        y_hat = train!(art, data.train_x[class_id], y=data.train_y[class_id])

        # Create a performances destination that is explicitly a vector of floats
        performances = zeros(Float64, n_data)
        for i = 1:n_data
            performances[i] = (y_hat[i] == data.train_y[class_id][i])
        end
    else
        error()
    end

    # Set the task parameters
    task_params = Dict("method" => "ddvfa")
    # Get a timestamp for the logging
    timestamp = DateTime(now())
    # Create a learning/evaluation block with the correct fields
    new_block = DataFrame(
        block_num = repeat([block_num], n_data),
        exp_num = collect(1+exp_num:n_data+exp_num),
        worker_id = repeat([0], n_data),
        block_type = repeat([mode], n_data),
        task_name = repeat([labels[class_id]], n_data),
        task_params = repeat([json(task_params)], n_data),
        exp_status = repeat(["complete"], n_data),
        timestamp = repeat([timestamp], n_data),
        performance = performances
    )
    # Write the dataframe to the file
    CSV.write(file, new_block, delim='\t')

    # Return the number of samples tested for the next run
    return n_data
end

# Test all of the classes
function test_all(exp_num::Integer, block_num::Integer)
    for i = 1:4
        # Make new experience directory
        experience_dir(args...) = scenario_dir("$block_num-test", args...)
        mkpath(experience_dir())
        file = experience_dir("data-log.tsv")
        n_data = run_one(art, flir_split, i, block_num, exp_num, file, "test")
        exp_num += n_data
        block_num += 1
    end
    return exp_num, block_num
end

# Train on one class
function train_one(exp_num::Integer, block_num::Integer, i::Integer)
    # Make new experience directory
    experience_dir(args...) = scenario_dir("$block_num-train", args...)
    mkpath(experience_dir())
    file = experience_dir("data-log.tsv")
    n_data = run_one(art, flir_split, i, block_num, exp_num, file, "train")
    exp_num += n_data
    block_num += 1
    return exp_num, block_num
end

# Run all of the training and testing
function run_all()
    # Iterate over all tests
    exp_num = 0
    block_num = 1
    # Test all of the classes first
    exp_num, block_num = test_all(exp_num, block_num)
    # Loop over each class
    for i = 1:4
        # Train on the class
        exp_num, block_num = train_one(exp_num, block_num, i)
        # Then test the performance of each class
        exp_num, block_num = test_all(exp_num, block_num)
    end
end

# -----------------------------------------
# SIMULATION
# -----------------------------------------

# Load the data
flir_split = FLIRL2MSplitCombined(data_dir, extractor)
# flir_split = FLIRL2MSplit(data_dir, extractor)

# Get the data stats
dim, _ = size(flir_split.train_x[1])

# Create the ART module
art = DDVFA(opts)

# Set the DDVFA config Manually
art.config = DataConfig(0.0, 1.0, dim)

# Run all the tests
run_all()
