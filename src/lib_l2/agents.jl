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
    Container for the Experience Queue
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
    # Use all of the DDVFA options for the logging parameters
    # for (key, value) in ddvfa_opts
    #     params[string(key)] = value
    # end
    for name in fieldnames(opts_DDVFA)
        params[string(name)] = getfield(ddvfa_opts, name)
    end

    # Construct and return the DDVFAAgent
    return DDVFAAgent(
        ddvfa,
        params,
        exp_container,
    )
end

"""
Constructor for a DDVFAAgent using the scenario dictionary and optional DDVFA keyword argument options.

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
Overload of the show function for DDVFAAgent.

# Arguments
- `io::IO`: the current IO stream.
- `cont::DDVFAAgent`: the DDVFAAgent to print/display.
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
Checks if the agent is done with its scenario queue.

# Arguments
- `agent::Agent`: the agent to test scenario completion on.
"""
function is_complete(agent::Agent)::Bool
    # Return a bool if the agent is complete with the scenario
    return (length(agent.scenario.queue) == 0)
end


"""
Evaluates a single agent on a single experience, training or testing as needed.

# Arguments
- `agent::Agent`: the agent to evaluate.
- `exp::Experience`: the experience to use for training/testing.
"""
function evaluate_agent(agent::Agent, experience::Experience, data::MatrixData)

    # # Sanitize the block type as train or test
    # sanitize_block_type(experience.block_type)

    # # if experience.block_type == "train":
    # if experience.update_model
    #     train!(agent.agent, data.train)
    # # elseif experience.block_type == "test":
    # else
    #     classify(agent.agent, )
    # end
    # agent.agent
    return
end

"""
Logs data from an L2 experience.

# Arguments
- `data_logger::PyObject`: the l2logger DataLogger.
- `exp::Experience`: the experience that the agent just processed.
- `results::Dict`: the results from the agent's experience.
- `status::AbstractString`: the if the experience was processed
"""
function log_data(data_logger::PyObject, experience::Experience, results::Dict, params::Dict ; status::AbstractString="complete")
    seq = experience.seq_nums
    worker = "9_l2metrics"
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
- `agent::Agent`: a struct that contains an `agent` and `scenario`.
- `data_logger::PyObject`: a l2logger object.
"""
function run_scenario(agent::Agent, data_logger::PyObject)
    # Load the data

    # Initialize the "last sequence"
    last_seq = SequenceNums(-1, -1)

    # Iterate while the agent's scenario is incomplete
    while !is_complete(agent)
        # Get the next experience
        exp = popfirst!(agent.scenario.queue)
        # Get the current sequence number
        cur_seq = exp.seq_nums
        # Logging
        if last_seq.block_num != cur_seq.block_num
            @info "New block: $(cur_seq.block_num)"
        end
        # Artificially create some results
        results = Dict(
            "performance" => 0.0,
        )
        # Log the data
        log_data(data_logger, exp, results, agent.params)

        # Loop reflection
        last_seq = cur_seq
    end

    return
end
