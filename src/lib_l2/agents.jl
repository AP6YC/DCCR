"""
    agents.jl

# Description
Definitions of agents and their evaluation.
"""

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------

"""
L2 agent supertype.
"""
abstract type Agent end

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
DDVFA-based L2 agent.
"""
struct DDVFAAgent <: Agent
    """
    The DDVFA module.
    """
    agent::DDVFA

    """
    Parameters used for l2logging.
    """
    params::Dict

    """
    Container for the Experience Queue.
    """
    scenario::ExperienceQueueContainer
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Creates a DDVFA agent with an empty experience queue.

# Arguments
- `ddvfa_opts::opts_DDVFA`: the options struct used to initialize the DDVFA module and set the logging params.
"""
function DDVFAAgent(ddvfa_opts::opts_DDVFA)
    # Create the DDVFA object from the opts
    ddvfa = DDVFA(ddvfa_opts)

    # Create the experience dequeue
    # exp_container = ExperienceQueueContainer(scenario_dict)
    exp_container = ExperienceQueueContainer()

    # Create the params object for Logging
    params = StatsDict()
    fields_to_dict!(params, ddvfa_opts)

    # Construct and return the DDVFAAgent
    return DDVFAAgent(
        ddvfa,
        params,
        exp_container,
    )
end

"""
Constructor for a [`DDVFAAgent`](@ref) using the scenario dictionary and optional DDVFA keyword argument options.

# Arguments
- `opts::AbstractDict`: keyword arguments for DDVFA options.
- `scenario::AbstractDict`: l2logger scenario as a dictionary.
"""
function DDVFAAgent(ddvfa_opts::opts_DDVFA, scenario_dict::AbstractDict)
    # Create an agent with an empty queue
    agent = DDVFAAgent(ddvfa_opts)
    # Initialize the agent's scenario container with the dictionary
    initialize_exp_queue!(agent.scenario, scenario_dict)
    # Return the agent with an initialized queue
    return agent
end

# -----------------------------------------------------------------------------
# TYPE OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`DDVFAAgent`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `cont::DDVFAAgent`: the [`DDVFAAgent`](@ref) to print/display.
"""
function Base.show(io::IO, agent::DDVFAAgent)
# function Base.show(io::IO, ::MIME"text/plain", agent::DDVFAAgent)
    # compact = get(io, :compact, false)
    print(io, "--- DDVFA agent with options:\n")
    # print(io, agent.agent.opts)
    print(io, agent.params)
    print(io, "\n--- Scenario: \n")
    print(io, agent.scenario)
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Checks if the [`Agent`](@ref) is done with its scenario queue.

# Arguments
- `agent::Agent`: the agent to test scenario completion on.
"""
function is_complete(agent::Agent)::Bool
    # Return a bool if the agent is complete with the scenario
    # return (length(agent.scenario.queue) == 0)
    return isempty(agent.scenario.queue)
end

"""
Gets an integer index of where a string name appears in a list of strings.

# Arguments
- `labels::Vector{T} where T <: AbstractString`: the list of strings to search.
- `name::AbstractString`: the name to search for in the list of labels.
"""
function get_index_from_name(labels::Vector{T}, name::AbstractString) where T <: AbstractString
    # Findall results in a list, even of only one entry
    results = findall(x -> x == name, labels)
    # If the list is longer than 1, error
    if length(results) > 1
        error("Labels list contains multiple instances of $name")
    end
    # If no error, return the first (and only) entry of the reverse search
    return results[1]
end

"""
Evaluates a single agent on a single experience, training or testing as needed.

# Arguments
- `agent::Agent`: the [`Agent`](@ref) to evaluate.
- `exp::Experience`: the [`Experience`](@ref) to use for training/testing.
"""
function evaluate_agent!(agent::Agent, experience::Experience, data::VectoredData)
    # Disect the experience
    dataset_index = get_index_from_name(data.train.labels, experience.task_name)
    datum_index = experience.seq_nums.task_num

    # If we are updating the model, run the training function
    if experience.update_model
        sample = data.train.x[dataset_index][:, datum_index]
        label = data.train.y[dataset_index][datum_index]
        y_hat = AdaptiveResonance.train!(agent.agent, sample, y=label)
    # elseif experience.block_type == "test":
    else
        sample = data.test.x[dataset_index][:, datum_index]
        label = data.test.y[dataset_index][datum_index]
        y_hat = AdaptiveResonance.classify(agent.agent, sample)
    end
    results = Dict(
        "performance" => y_hat == label ? 1.0 : 0.0,
        "art_match" => agent.agent.stats["M"],
        "art_activation" => agent.agent.stats["T"],
    )
    # agent.agent
    # # Artificially create some results
    # results = Dict(
    #     "performance" => 0.0,
    # )
    return results
end

"""
Logs data from an L2 [`Experience`](@ref).

# Arguments
- `data_logger::PyObject`: the l2logger DataLogger.
- `exp::Experience`: the [`Experience`](@ref) that the [`Agent`](@ref) just processed.
- `results::Dict`: the results from the [`Agent`](@ref)'s [`Experience`](@ref).
- `status::AbstractString`: string expressing if the experience was processed.
"""
function log_data(data_logger::PyObject, experience::Experience, results::Dict, params::Dict ; status::AbstractString="complete")
    seq = experience.seq_nums
    worker = "l2metrics"
    record = Dict(
        "block_num" => seq.block_num,
        "block_type" => experience.block_type,
        # "task_params" => exp.params,
        "task_params" => params,
        "task_name" => experience.task_name,
        "exp_num" => seq.exp_num,
        "exp_status" => status,
        "worker_id" => worker,
    )
    merge!(record, results)
    data_logger.log_record(record)
end

"""
Runs an agent's scenario.

# Arguments
- `agent::Agent`: a struct that contains an [`Agent`](@ref) and `scenario`.
- `data_logger::PyObject`: a l2logger object.
"""
function run_scenario(agent::Agent, data::VectoredData, data_logger::PyObject)
    # Initialize the "last sequence"
    # last_seq = SequenceNums(-1, -1, -1)

    # Initialize the progressbar
    n_exp = length(agent.scenario.queue)
    # block_log_string = "Block 1"
    p = Progress(n_exp; showspeed=true)
    # Iterate while the agent's scenario is incomplete
    while !is_complete(agent)
        # Get the next experience
        exp = popfirst!(agent.scenario.queue)
        # Get the current sequence number
        # cur_seq = exp.seq_nums
        # Logging
        next!(p; showvalues = [
            # (:Block, cur_seq.block_num),
            (:Block, exp.seq_nums.block_num),
            (:Type, exp.block_type),
        ])
        # Evaluate the agent on the experience
        results = evaluate_agent!(agent, exp, data)

        # Log the data
        log_data(data_logger, exp, results, agent.params)

        # Loop reflection
        # last_seq = cur_seq
    end

    return
end
