"""
    lib_l2.jl

# Description
A collection of l2-specific experiment function and struct definitions.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

using
    JSON

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

# Pretty indentation in JSON files
JSON_INDENT = 4

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

# Taken from l2logger_template
mutable struct SequenceNums
    block_num::Int
    exp_num::Int
end

# Taken from l2logger_template
mutable struct Experience
    agent::DDVFA
    task_name::String
    seq_nums::SequenceNums
    block_type::String
    update_model::Bool
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

function log_data(data_logger, exp::Experience, results::Dict, status::String="completed")
    seq = exp.sequence_nums
    worker = "9_l2metrics"
    record = Dict(
        "block_num" => seq.block_num,
        "block_type" => exp.block_type,
        "task_params" => exp.params,
        "task_name" => exp.task_name,
        "exp_num" => seq.exp_num,
        "exp_status" => status,
        "worker_id" => worker,
    )
    merge!(record, results)
    data_logger.log_record(record)
end

"""
Saves the dictionary to a JSON file.

# Arguments
- `filepath::AbstractString`: the full file name (with path) to save to.
- `dict::AbstractDict`: the dictionary to save to the file.
"""
function json_save(filepath::AbstractString, dict::AbstractDict)
    # Use the with open syntax to print directly to the file.
    open(filepath, "w") do f
        JSON.print(f, dict, JSON_INDENT)
    end
end

"""
Loads the JSON file, interpreted as a dictionary.

# Arguments
- `filepath::AbstractString`: the full file name (with path) to load.
"""
function json_load(filepath::AbstractString)
    return JSON.parsefile(filepath)
end
