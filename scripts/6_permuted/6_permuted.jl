"""
    6_permuted.jl

# Description
Because each process does a lot of work, `pmap` is used,
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
using Combinatorics     # permutations

# -----------------------------------------------------------------------------
# PARSE ARGS
# -----------------------------------------------------------------------------

pargs = DCCR.dist_exp_parse(
    "6_permuted: distributed permuted condensed scenarios."
)

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Start several processes
if pargs["procs"] > 0
    addprocs(pargs["procs"], exeflags="--project=.")
end

# Set the simulation parameters
orders = collect(1:6)
sim_params = Dict{String, Any}(
    "m" => "ddvfa",
    "order" => collect(permutations(orders))
    # "seed" => collect(1:1000)
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
    experiment_top = "6_permuted"

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
    data = DCCR.load_orbits(data_dir, data_dirs, opts["opts_DDVFA"])

    # Sort/reload the data as indexed components
    data_indexed = DCCR.get_indexed_data(data)

    # Define a single-parameter function for pmap
    local_sim(dict) = DCCR.permuted(dict, data_indexed, opts["opts_DDVFA"])
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
