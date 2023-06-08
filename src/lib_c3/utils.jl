"""
    utils.jl

# Description
This file contains utilities for file handling and results saving.
"""

# Results directories (local and paper)
results_dir(args...) = projectdir("work", "results", args...)
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

const SAVE_MAP = Dict(
    "figure" => :_save_dccr_fig,
    "table" => :_save_dccr_table,
)

"""
Figure saving function
"""
function save_dccr(type::AbstractString, object, exp_name::AbstractString, fig_name::AbstractString ; to_paper::Bool=false)
    # Save the figure to the local results directory
    mkpath(results_dir(exp_name))
    eval(SAVE_MAP[type])(object, results_dir(exp_name, fig_name))

    # Check if saving to the paper directory as well
    if to_paper
        mkpath(paper_results_dir(exp_name))
        eval(SAVE_MAP[type])(object, paper_results_dir(exp_name, fig_name))
    end
end



# results_dir(args...) = projectdir("work", "results", args...)
# paper_results_dir(args...) = joinpath("C:\\", "Users", "Sasha", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3-V3-Resubmission", "images", "results", args...)
# # paper_results_dir(args...) = joinpath("C:\\", "Users", "sap62", "Dropbox", "Apps", "Overleaf", "Paper-MST-TDY-C3-V3-Resubmission", "images", "results", experiment_top, args...)

# function s

# function save_dccr_fig(args... ; to_paper::Bool=false)

# end