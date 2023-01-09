# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

# Pretty indentation in JSON files
const JSON_INDENT = 4

# Valid types of certain options
const BLOCK_TYPES = ["train", "test"]
const LOG_STATES = ["complete", "incomplete"]

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Sanitizes a selection within a list of acceptable options.

# Arguments
- `selection_type::AbstractString`: a string describing the option in case it is misused.
- `selection::Any`: a single selection from a list
"""
function sanitize_in_list(selection_type::AbstractString, selection::T, acceptable::Vector{T}) where T <: Any
    # Verify that we have a correct selection
    try
        @assert selection in acceptable
    catch
        error("$(selection_type) must be one of the following: $(acceptable)")
    end
end

"""
Sanitize the selected block type against the BLOCK_TYPES constant.

# Arguments
- `block_type::AbstractString`: the selected block type.
"""
function sanitize_block_type(block_type::AbstractString)
    # Verify that we have a correct block type
    sanitize_in_list("block_type", block_type, BLOCK_TYPES)
end

"""
Sanitize the selected log state against the LOG_STATES constant.

# Arguments
- `log_state::AbstractString`: the selected log state.
"""
function sanitize_log_state(log_state::AbstractString)
    # Verify that we have a correct log state
    sanitize_in_list("log_state", log_state, LOG_STATES)
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
