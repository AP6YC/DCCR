"""
    3_driver.jl

# Description
Runs the l2 condensed scenario specified by the gen_scenario.jl file.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DCCR

# -----------------------------------------------------------------------------
# ADDITIONAL DEPENDENCIES
# -----------------------------------------------------------------------------

using AdaptiveResonance

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "9_l2metrics"

# DCCR project files
# include(projectdir("src", "setup.jl"))

# Special l2 setup for this experiment (setting the pyenv, etc.)
include(DCCR.projectdir("src", "setup_l2.jl"))

# Load the l2logger PyCall dependency
l2logger = pyimport("l2logger.l2logger")

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
