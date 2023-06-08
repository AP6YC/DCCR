"""
    utils.jl

# Description
This file contains utilities for file handling and results saving.
"""

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
