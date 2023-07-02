"""
    test_sets.jl

# Description
The main collection of tests for the DCCR.jl package.
This file loads common utilities and aggregates all other unit tests files.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    Logging,
    DCCR,
    Test

@testset "Boilerplate" begin
    @assert 1 == 1
end

@testset "DCCR" begin
    # Parse arguments
    pargs = DCCR.exp_parse(
        "1_test: unit test."
    )

    # Simulation options
    opts_file = "default.yml"
    # @info DCCR.opts_file(opts_file)
    @info DCCR.projectdir("opts", opts_file)
    @info isfile(DCCR.projectdir("opts", opts_file))
    @info readdir(DCCR.projectdir())
    @info readdir(pwd())
    # # Load the default simulation options
    # opts = DCCR.load_sim_opts(opts_file)

    # # Load the data names and class labels from the selection
    # data_dirs, class_labels = DCCR.get_orbit_names(opts["data_selection"])

end
