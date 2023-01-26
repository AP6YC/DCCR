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

# if !haskey(ENV, "PYTHON")
#     PYTHON_ENV = raw"C:\Users\Sasha\Anaconda3\envs\l2mmetrics\python.exe"
#     if isfile(PYTHON_ENV)
#         ENV["PYTHON"] = PYTHON_ENV
#     else
#         ENV["PYTHON"] = ""
#     end
# end

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
data, data_indexed, class_labels, data_selection, n_classes = load_default_orbit_data(data_dir)

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
run_scenario(agent, data_indexed, data_logger)
