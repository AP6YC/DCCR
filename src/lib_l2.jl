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
    DataStructures,     # Dequeue
    JSON                # JSON file load/save

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

# Pretty indentation in JSON files
const JSON_INDENT = 4

# Valid block types
const BLOCK_TYPES = ["train", "test"]

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
Sequence numbers for a block and experience.

Taken from l2logger_template.
"""
struct SequenceNums
    block_num::Int
    exp_num::Int
end

"""
Experience block for an agent.

Taken from l2logger_template.
"""
struct Experience
    task_name::String
    seq_nums::SequenceNums
    block_type::String
    update_model::Bool
end

"""
Alias for a queue of Experiences.
"""
const ExperienceQueue = Deque{Experience}

"""
Container for the ExperienceQueue and some statistics about it.
"""
struct ExperienceQueueContainer
    queue::ExperienceQueue
    stats::Dict{String, Any}
end

"""
Creates a queue of Experiences from the scenario dictionary.

# Arguments
- `scenario_dict::AbstractDict`: the scenario dictionary.
"""
function ExperienceQueueContainer(scenario_dict::AbstractDict)
    # Create an empty statistics container
    stats = Dict{String, Any}(
        "length" => 0,
        "n_blocks" => 0,
        "n_train" => 0,
        "n_test" => 0,
    )

    # Create an empty experience Deque
    exp_queue = Deque{Experience}()
    # Start the experience count
    exp_num = 0
    # Start the block count
    block_num = 0
    # Iterate over each learning/testing block
    for block in scenario_dict["scenario"]
        # Increment the block count
        block_num += 1
        # Get the block type as "train" or "test"
        block_type = block["type"]
        # Verify the block type
        check_block_type(block_type)
        # Stats on the blocks
        if block_type == "train"
            stats["n_train"] += 1
        elseif block_type == "test"
            stats["n_test"] += 1
        end
        # Iterate over the regimes of the block
        for regime in block["regimes"]
            # Iterate over each count within the current regime
            for _ in 1:regime["count"]
                # Increment the experience count
                exp_num += 1
                # Get the task name
                task_name = regime["task"]
                # Create a sequence number container for the block and experience
                seq = SequenceNums(block_num, exp_num)
                # Create an experience for all of the above
                exp = Experience(task_name, seq, block_type)
                # Add the experience to the top of the Deque
                push!(exp_queue, exp)
            end
        end
    end

    # Post processing stats
    stats["length"] = length(exp_queue)
    stats["n_blocks"] = block_num

    # Return the a container with the experiences
    return ExperienceQueueContainer(
        exp_queue,
        stats,
    )
end

"""
L2 agent supertype.
"""
abstract type Agent end

"""
DDVFA-based L2 agent.
"""
struct DDVFAAgent <: Agent
    agent::DDVFA
    scenario::ExperienceQueueContainer
end

"""
Overload of the show function for ExperienceQueueContainer.
"""
function Base.show(io::IO, cont::ExperienceQueueContainer)
    # compact = get(io, :compact, false)
    print(
        io,
        """
        ExperienceQueueContainer
        ExperienceQueue of type $(ExperienceQueue)
        Length: $(cont.stats["length"])
        Blocks: $(cont.stats["n_blocks"])
        Train blocks: $(cont.stats["n_train"])
        Test blocks: $(cont.stats["n_test"])
        """
    )
end

"""
Overload of the show function for ExperienceQueue.
"""
function Base.show(io::IO, queue::ExperienceQueue)
    # compact = get(io, :compact, false)
    print(
        io,
        """
        ExperienceQueue of type $(ExperienceQueue)
        Length: $(length(queue))
        """
    )
end

"""
Overload of the show function for DDVFAAgent.
"""
function Base.show(io::IO, agent::DDVFAAgent)
# function Base.show(io::IO, ::MIME"text/plain", agent::DDVFAAgent)
    # compact = get(io, :compact, false)
    print(io, "DDVFA agent with options:\n")
    print(io, agent.agent.opts)
    print(io, "Scenario: \n")
    print(io, agent.scenario)
end

# -----------------------------------------------------------------------------
# CONSTRUCTOR METHODS
# -----------------------------------------------------------------------------

function check_block_type(block_type::AbstractString)
    # Verify that we have a correct block type
    try
        @assert block_type in BLOCK_TYPES
    catch
        error("block_type must be one of the following: $(BLOCK_TYPES)")
    end
end

"""
Constructs an Experience, setting the update_model field based upon the block type.

# Arguments
- `task_name::AbstractString`: the name of the current task.
- `seq_nums::SequenceNums`: the block and experience number of the experience.
- `block_type::AbstractString`: the block type ∈ ["train", "test"]. Using "train" sets update_model to true, "test" to false.
"""
function Experience(task_name::AbstractString, seq_nums::SequenceNums, block_type::AbstractString)
    # Verify the block type
    check_block_type(block_type)

    # Construct and return the Experience
    return Experience(
        task_name,
        seq_nums,
        block_type,
        block_type == "train",
    )
end

"""
Constructor for a DDVFAAgent using the scenario dictionary and optional DDVFA keyword argument options.

# Arguments
- `scenario::AbstractDict`: l2logger scenario as a dictionary.
- `kwargs...`: keyword arguments for DDVFA options.
"""
function DDVFAAgent(scenario_dict::AbstractDict ; kwargs...)
    # Create the DDVFA options from the kwargs
    opts = opts_DDVFA(;kwargs...)

    # Create the DDVFA object from the opts
    ddvfa = DDVFA(opts)

    # Create the experience dequeue
    exp_container = ExperienceQueueContainer(scenario_dict)

    # Construct and return the DDVFAAgent
    return DDVFAAgent(
        ddvfa,
        exp_container,
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""

# Arguments
- `exp::Experience`:
- `results::Dict`:
- `status::AbstractString`:
"""
function log_data(data_logger, exp::Experience, results::Dict, params::Dict ; status::AbstractString="completed")
    seq = exp.seq_nums
    worker = "9_l2metrics"
    record = Dict(
        "block_num" => seq.block_num,
        "block_type" => exp.block_type,
        # "task_params" => exp.params,
        # "task_params" => params,
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

"""
Checks if the agent is done with its scenario queue.

# Arguments
- `agent::Agent`: the agent to test scenario completion on.
"""
function is_complete(agent::Agent)
    return (length(agent.scenario.queue) == 0)
end

"""
Runs an agent's scenario.

# Arguments
- `agent::Agent`: a struct that contains an `agent` and `scenario`.
- `data_logger`: a l2logger object.
"""
function run_scenario(agent::Agent, data_logger)
    # for experience in agent.scenario
    # Initialize the "last sequence"
    last_seq = SequenceNums(-1, -1)
    # Iterate while the agent's scenario is incomplete
    while !is_complete(agent)
        # Get the next experience
        exp = popfirst!(agent.scenario.queue)
        # Get the current sequence number
        cur_seq = exp.seq_nums
        if last_seq.block_num != cur_seq.block_num
            @info "New block: $(cur_seq.block_num)"
        end
        results = Dict(
            "performance" => 0.0,
        )
        log_data(data_logger, exp, results)
    end

    return
end