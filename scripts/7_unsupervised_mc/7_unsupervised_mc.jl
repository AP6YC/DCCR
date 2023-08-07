"""
    7_unsupervised_mc.jl

# Description
This script runs an unsupervised learning scenario. After supervised pretraining,
the module learns upon additional data unsupervised and is tested for performance
before and after. This is done as a monte carlo of many simulations.

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

using Distributed

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

pargs = DCCR.dist_exp_parse(
    "3_shuffled_mc: distributed simple shuffled train/test Monte Carlo."
)

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Start several processes
if pargs["procs"] > 0
    addprocs(pargs["procs"], exeflags="--project=.")
end

# Set the simulation parameters
sim_params = Dict{String, Any}(
    "m" => "ddvfa",
    "seed" => collect(1:1000)
)

# -----------------------------------------------------------------------------
# PARALLEL DEFINITIONS
# -----------------------------------------------------------------------------

@everywhere begin
    # Activate the project in case
    using Pkg
    Pkg.activate(".")

    # Modules
    using Revise            # Editing this file
    using DCCR

    # Experiment save directory name
    experiment_top = "7_unsupervised_mc"

    # Make a path locally just for the sweep results
    sweep_results_dir(args...) = DCCR.unpacked_dir(experiment_top, "sweep", args...)
    mkpath(sweep_results_dir())

    # Load the default simulation options
    opts = DCCR.load_sim_opts()

    # Load the data names and class labels from the selection
    data_dirs, class_labels = DCCR.get_orbit_names(opts["data_selection"])

    # Number of classes
    n_classes = length(data_dirs)

    # Load the orbits
    @info "Worker $(myid()): loading data"
    data = DCCR.load_orbits(data_dir, data_dirs, opts["scaling"])

    # Create a combine dataset for unsupervised_mc
    combined_data = DCCR.DataSplitCombined(data)

    # Define a single-parameter function for pmap
    local_sim(dict) = DCCR.unsupervised_mc(dict, combined_data, opts["opts_DDVFA"])
end

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

# Log the simulation scale
@info "START: $(dict_list_count(sim_params)) simulations across $(nprocs())."

# Turn the dictionary of lists into a list of dictionaries
dicts = dict_list(sim_params)

# Remove impermissible sim options
# filter!(d -> d["rho_ub"] > d["rho_lb"], dicts)
# @info "Testing permutations:" dicts

# Parallel map the sims
pmap(local_sim, dicts)

# Save the data into a binary
pack_data(experiment_top)

println("--- Simulation complete ---")

# -----------------------------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------------------------

# Close the workers after simulation
if pargs["procs"] > 0
    rmprocs(workers())
end
