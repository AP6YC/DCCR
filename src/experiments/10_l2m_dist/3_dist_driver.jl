"""
    3_dist_driver.jl

# Description
Runs the l2 condensed scenario specified by the gen_scenario.jl file.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

using
    Distributed,
    Combinatorics

# addprocs(24, exeflags="--project=.")
# If we parallelize from the command line
if !isempty(ARGS)
    addprocs(parse(Int, ARGS[1]), exeflags="--project=.")
end

@everywhere begin
    # Load dependencies
    using
        DrWatson,
        AdaptiveResonance

    # Experiment save directory name
    experiment_top = "10_l2m_dist"

    # DCCR project files
    include(projectdir("src", "setup.jl"))

    # Special l2 setup for this experiment (setting the pyenv, etc.)
    include(projectdir("src", "setup_l2.jl"))

    # Load the l2logger PyCall dependency
    l2logger = pyimport("l2logger.l2logger")

    # -----------------------------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------------------------

    # Load the default data configuration
    data, data_indexed, class_labels, data_selection, n_classes = load_default_orbit_data(data_dir)

    # -----------------------------------------------------------------------------
    # EXPERIMENT
    # -----------------------------------------------------------------------------

    function run_scenario_permuation(
        order::Vector{Int},
        # config_dir_perm::AbstractString,
        data_indexed::VectoredData,
    )
        text_order = String(join(order))
        # Load the config and scenario
        config = json_load(configs_dir(text_order, "config.json"))
        scenario = json_load(configs_dir(text_order, "scenario.json"))

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
    end

    local_sim(order) = run_scenario_permuation(order, data_indexed)

end

# Get a list of the order indices
orders = collect(1:6)

# Create an iterator for all permutations and make it into a list
orders = collect(permutations(orders))

# Parallel map the sims
pmap(local_sim, orders)

# Save the data into a binary
# pack_data(experiment_top)

println("--- Simulation complete ---")

# Close the workers after simulation
if !isempty(ARGS)
    rmprocs(workers())
end
