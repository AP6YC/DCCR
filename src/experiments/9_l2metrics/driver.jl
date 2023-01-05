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
# EXPERIMENT
# -----------------------------------------------------------------------------

# Setup the scenario_info dictionary as a function of the config and scenario
scenario_info = config["META"]
scenario_info["input_file"] = scenario

# Instantiate the data logger
data_logger = l2logger.DataLogger(
    config["NAME"],
    config["DIR"],
    config["COLS"],
    scenario_info,
)

# SCENARIO_DIR = "simple"
# SCENARIO_INFO = Dict(
#     "author" => "Sasha Petrenko",
#     "complexity" => "1-low",
#     "difficulty" => "2-medium",
#     "scenario_type" => "custom",
# )
# LOGGER_INFO = Dict(
#     # "metrics_columns" => "reward",
#     "metrics_columns" => "performance",
#     "log_format_version" => "1.0",
# )
