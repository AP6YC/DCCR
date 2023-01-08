"""
    driver.jl

# Description
Runs the l2 condensed scenario specified by the gen_scenario.jl file.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Load the default environment
ENV["PYTHON"] = ""

# Load dependencies
using
    PyCall,
    DrWatson,
    AdaptiveResonance

# Load the l2logger PyCall dependency
l2logger = pyimport("l2logger.l2logger")

# Experiment save directory name
experiment_top = "9_l2metrics"

# DCCR project files
include(projectdir("src", "setup.jl"))

# Special folders for this experiment
include(projectdir("src", "setup_l2.jl"))

# Load the config and scenario
config = json_load(configs_dir("config.json"))
scenario = json_load(configs_dir("scenario.json"))

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

# Load the default data configuration
data, data_indexed, class_labels, n_classes = load_default_orbit_data(data_dir)

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

# Setup the scenario_info dictionary as a function of the config and scenario
scenario_info = config["META"]
scenario_info["input_file"] = scenario

# Instantiate the data logger
data_logger = l2logger.DataLogger(
    config["DIR"],
    config["NAME"],
    config["COLS"],
    scenario_info,
)

# Create the DDVFA options for both initialization and logging
ddvfa_opts = opts_DDVFA(
    # DDVFA options
    gamma = 5.0,
    gamma_ref = 1.0,
    # rho=0.45,
    rho_lb = 0.45,
    rho_ub = 0.7,
    similarity = :single,
    display = false,
)
# Construct the agent from the scenario
agent = DDVFAAgent(
    ddvfa_opts,
    scenario,
)

# Specify the input data configuration
agent.agent.config = DataConfig(0, 1, 128)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Run the scenario
run_scenario(agent, data_logger)

# # Get the data dimensions
# dim, n_train = size(data.train.x)
# _, n_test = size(data.test_x)

# Create the estimate containers
# perfs = [[] for i = 1:n_classes]

# vals = [[] for i = 1:n_classes]

# # Initial testing block
# for j = 1:n_classes
#     push!(perfs[j], 0.0)
# end

# vals = []
# test_interval = 20

# # Iterate over each class
# for i = 1:n_classes
#     # Learning block
#     _, n_samples_local = size(data_indexed.train_x[i])
#     # local_vals = zeros(n_classes, n_samples_local)
#     # local_vals = zeros(n_classes, 0)
#     local_vals = Array{Float64}(undef, n_classes, 0)

#     # Iterate over all samples
#     @showprogress for j = 1:n_samples_local
#         train!(ddvfa, data_indexed.train_x[i][:, j], y=data_indexed.train_y[i][j])

#         # Validation intervals
#         if j % test_interval == 0
#             # Validation data
#             # local_y_hat = AdaptiveResonance.classify(ddvfa, data.val_x, get_bmu=true)
#             # local_val = get_accuracies(data.val_y, local_y_hat, n_classes)
#             # Training data
#             local_y_hat = AdaptiveResonance.classify(ddvfa, data.train.x, get_bmu=true)
#             local_val = get_accuracies(data.train.y, local_y_hat, n_classes)
#             local_vals = hcat(local_vals, local_val')
#         end
#     end

#     push!(vals, local_vals)

#     # Experience block
#     for j = 1:n_classes
#         local_y_hat = AdaptiveResonance.classify(ddvfa, data_indexed.test_x[j], get_bmu=true)
#         push!(perfs[j], performance(local_y_hat, data_indexed.test_y[j]))
#     end
# end

# # Clean the NaN vals
# for i = 1:n_classes
#     replace!(vals[i], NaN => 0.0)
# end