"""
    experience.jl

# Description
Definitions of what individual l2 experiences are.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
Sequence numbers for a block and experience.

Taken from l2logger_template.
"""
struct SequenceNums
    """
    The block number.
    """
    block_num::Int

    """
    The experience number.
    """
    exp_num::Int

    """
    The task-specific count.
    """
    task_num::Int
end

"""
Experience block for an agent.

Taken from l2logger_template.
"""
struct Experience
    """
    The task name.
    """
    task_name::String

    """
    The sequence numbers (block and experience count).
    """
    seq_nums::SequenceNums

    """
    The block type, valid values are ∈ $(BLOCK_TYPES).
    """
    block_type::String

    """
    Flag for updating the model (i.e., true is to train, false is to classify).
    """
    update_model::Bool
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Constructs an Experience, setting the update_model field based upon the block type.

# Arguments
- `task_name::AbstractString`: the name of the current task.
- `seq_nums::SequenceNums`: the block and experience number of the experience.
- `block_type::AbstractString`: the block type ∈ ["train", "test"]. Using "train" sets update_model to true, "test" to false.
"""
function Experience(task_name::AbstractString, seq_nums::SequenceNums, block_type::AbstractString)
    # Verify the block type
    sanitize_block_type(block_type)

    # Construct and return the Experience
    return Experience(
        task_name,
        seq_nums,
        block_type,
        block_type == "train",
    )
end
