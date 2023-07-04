"""
    utils.jl

# Description
This file contains utilities for file handling and results saving.
"""

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
Common doc: directory string to save to.
"""
const DOC_ARG_SAVE_DIR = """
- `dir::AbstractString`: the directory to save to.
"""

"""
Wrapper for how figures are saved in the DCCR project.

# Arguments
- `fig`: the figure object to save.
$DOC_ARG_SAVE_DIR
"""
function _save_dccr_fig(fig, dir::AbstractString)
    savefig(fig, dir)
end

"""
Wrapper for how tables are saved in the DCCR project.

# Arguments
- `table`: the table object to save.
$DOC_ARG_SAVE_DIR
"""
function _save_dccr_table(table, dir::AbstractString)
    open(dir, "w") do io
        write(io, table)
    end
end

"""
Dictionary mapping the names of result save types to the private wrapper functions that implement them.
"""
const SAVE_MAP = Dict(
    "figure" => :_save_dccr_fig,
    "table" => :_save_dccr_table,
)

"""
Type alias for how parsed arguments are treated.
"""
const ParsedArgs = Dict{String, Any}

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# Results directories (local and paper)
"""
`DrWatson`-style local results directory.
"""
results_dir(args...) = projectdir("work", "results", args...)

"""
`DrWatson`-style paper results directory.
"""
paper_results_dir(args...) = joinpath("C:\\", "Users", "Sasha", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3-V3-Resubmission", "images", "results", args...)
# paper_results_dir(args...) = joinpath("C:\\", "Users", "sap62", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3-V3-Resubmission", "images", "results", experiment_top, args...)

"""
Saving function for results in the DCCR project.

This function dispatches to the correct private wrapper saving function via the `type` option, and the `to_paper` flag determines if the result is also saved to a secondary location, which is mainly used for also saving the result to the cloud location for the journal paper.

# Arguments
- `type::AbstractString`: the type of object being saved (see [`SAVE_MAP`](@ref)).
- `object`: the object to save as `type`, whether a figure, table, or something else.
- `exp_name::AbstractString`: the name of the experiment, used for the final saving directories.
- `save_name::AbstractString`: the name of the save file itself.
- `to_paper::Bool=false`: optional, flag for saving to the paper results directory (default `false`).
"""
function save_dccr(type::AbstractString, object, exp_name::AbstractString, save_name::AbstractString ; to_paper::Bool=false)
    # Save the figure to the local results directory
    mkpath(results_dir(exp_name))
    eval(SAVE_MAP[type])(object, results_dir(exp_name, save_name))

    # Check if saving to the paper directory as well
    if to_paper
        mkpath(paper_results_dir(exp_name))
        eval(SAVE_MAP[type])(object, paper_results_dir(exp_name, save_name))
    end
end

const ARG_ARG_DESCRIPTION = """
# Arguments
- `description::AbstractString`: optional positional, the script description for the parser
"""

"""
Common function for how `ArgParseSettings` are generated in the project.

$ARG_ARG_DESCRIPTION
"""
function get_argparsesettings(description::AbstractString="")
    # Set up the parse settings
    s = ArgParseSettings(
        description = description,
        commands_are_required = false,
        version = string(DCCR_VERSION),
        add_version = true
    )
    return s
end

"""
Parses the command line for common options in serial (non-distributed) experiments.

$ARG_ARG_DESCRIPTION
"""
function exp_parse(description::AbstractString="A DCCR experiment script.")
    # Set up the parse settings
    s = get_argparsesettings(description)

    # Set up the arguments table
    @add_arg_table! s begin
        "--paper", "-p"
            help = "flag for saving results to the paper directory"
            action = :store_true
        "--no-display", "-d"
            help = "flag for running headless, suppressing the display of generated figures"
            action = :store_true
        "--verbose", "-v"
            help = "flag for verbose output"
            action = :store_true
    end

    # Parse and return the arguments
    return parse_args(s)
end

"""
Parses the command line for common options in distributed experiments.

$ARG_ARG_DESCRIPTION
"""
function dist_exp_parse(description::AbstractString="A distributed DCCR experiment script.")
    # Set up the parse settings
    s = get_argparsesettings(description)

    # Set up the arguments table
    @add_arg_table! s begin
        "--procs", "-p"
            help = "number of parallel processes"
            arg_type = Int
            default = 0
        "--n_sims", "-n"
            help = "the number of simulations to run"
            arg_type = Int
            default = 1
        "--verbose", "-v"
            help = "verbose output"
            action = :store_true
    end

    # Parse and return the arguments
    return parse_args(s)
end

"""
Loads the provided options YAML file.

# Arguments
- `file::AbstractString`: the YAML file to load.
"""
function load_opts(file::AbstractString)
    # Point to the default location of the file
    full_path = projectdir("opts", file)
    # Load the YAML options file as a string-keyed dictionary
    file_opts = YAML.load_file(full_path, dicttype=Dict{String, Any})
    # Return the dictionary
    return file_opts
end

"""
Loads and returns the simulation options from the provided YAML file.

# Arguments
- `file::AbstractString="default.yml"`: options the file to load from the options directory.
"""
function load_sim_opts(file::AbstractString="default.yml")
    # Load the options dict
    opts_dict = load_opts(file)

    # Parse the DDVFA options
    dd = opts_dict["opts_DDVFA"]
    opts = opts_DDVFA(
        gamma = dd["gamma"],
        gamma_ref = dd["gamma_ref"],
        rho_lb = dd["rho_lb"],
        rho_ub = dd["rho_ub"],
        similarity = Symbol(dd["similarity"]),
    )

    # Overwrite the dictionary entry with the actual options
    opts_dict["opts_DDVFA"] = opts

    # Return the simulation options dictionary
    return opts_dict
end

"""
Handles the display of plots according to arguments parsed by the script.

# Arguments
- `p::Plots.Plot`: the plot handle to display if necessary.
- `pargs::ParsedArgs`: the parsed arguments provided by the script.
"""
function handle_display(p::Plots.Plot, pargs::ParsedArgs)
    # Display if the flag is low
    !pargs["no-display"] && display(p)
end

"""
Wrapper for loading simulation results with arbitrarily many fields.

# Arguments
- `data_file::AbstractString`: the location of the datafile for loading.
- `args...`: the string names of the files to open.
"""
function load_sim_results(data_file::AbstractString, args...)
    # Load and return the tuple of entries from the data file
    return JLD2.load(data_file, args...)
end

# """
# Wrapper for saving simulation results with arbitrarily many fields.

# # Arguments
# - `data_file::AbstractString`: the location of the datafile for saving.
# - `args...`: the variables to save.
# """
# function save_sim_results(data_file::AbstractString, args...)
#     # Save the data
#     jldsave(data_file; args...)
# end
