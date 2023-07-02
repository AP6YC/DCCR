"""
    3_shuffled_mc.jl

# Description
This script runs a Monte Carlo simulation of the simple shuffled train/test
scenario in parallel. Because each process does a lot of work, `pmap`` is used,
requiring every process to be spawned ahead of time and passed the necessary
function definitions to run each simulation.

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
    "seed" => collect(1:pargs["n_sims"]),
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
    experiment_top = "3_shuffled_mc"

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

    # Define a single-parameter function for pmap
    local_sim(dict) = DCCR.shuffled_mc(dict, data, opts["opts_DDVFA"])
end

# Log the simulation scale
@info "START: $(dict_list_count(sim_params)) simulations across $(nprocs())."

# Turn the dictionary of lists into a list of dictionaries
dicts = dict_list(sim_params)

# Remove impermissible sim options
# filter!(d -> d["rho_ub"] > d["rho_lb"], dicts)
# @info "Testing permutations:" dicts

# Parallel map the sims
pmap(local_sim, dicts)

# Close the workers after simulation
if pargs["procs"] > 0
    rmprocs(workers())
end
