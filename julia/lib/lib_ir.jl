using StatsBase
using AdaptiveResonance
using Statistics
using Logging
using HDF5              # Loading .h5 activation files

"""
    DataSplit

A basic struct for encapsulating the four components of supervised training.
"""
mutable struct DataSplit
    train_x::Array
    test_x::Array
    train_y::Array
    test_y::Array
    DataSplit(train_x, test_x, train_y, test_y) = new(train_x, test_x, train_y, test_y)
end

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Float)

Return a DataSplit struct that is split by the ratio (e.g. 0.8).
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real)
    dim, n_data = size(data_x)
    split_ind = Int(floor(n_data * ratio))

    train_x = data_x[:, 1:split_ind]
    test_x = data_x[:, split_ind + 1:end]
    train_y = data_y[1:split_ind]
    test_y = data_y[split_ind + 1:end]

    return DataSplit(train_x, test_x, train_y, test_y)
end

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

Sequential loading and ratio split of the data.
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)
    dim, n_data = size(data_x)
    n_splits = length(seq_ind)

    train_x = Array{Float64}(undef, dim, 0)
    train_y = Array{Float64}(undef, 0)
    test_x = Array{Float64}(undef, dim, 0)
    test_y = Array{Float64}(undef, 0)

    # Iterate over all splits
    for ind in seq_ind
        local_x = data_x[:, ind[1]:ind[2]]
        local_y = data_y[ind[1]:ind[2]]
        # n_data = ind[2] - ind[1] + 1
        n_data = size(local_x)[2]
        split_ind = Int(floor(n_data * ratio))

        train_x = [train_x local_x[:, 1:split_ind]]
        test_x = [test_x local_x[:, split_ind + 1:end]]
        train_y = [train_y; local_y[1:split_ind]]
        test_y = [test_y; local_y[split_ind + 1:end]]
    end
    return DataSplit(train_x, test_x, train_y, test_y)
end # DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

"""
    sigmoid(x::Real)

Return the sigmoid function on x.
"""
function sigmoid(x::Real)
# return 1.0 / (1.0 + exp(-x))
    return one(x) / (one(x) + exp(-x))
end

"""
feature_preprocess!(data_split::DataSplit)
"""
function feature_preprocess!(data_split::DataSplit)
    # Standardize
    dt_train = fit(ZScoreTransform, data_split.train_x, dims=2)
    dt_test = fit(ZScoreTransform, data_split.test_x, dims=2)
    data_split.train_x = StatsBase.transform(dt_train, data_split.train_x)
    data_split.test_x = StatsBase.transform(dt_test, data_split.test_x)

    # Squash the data sigmoidally in case of outliers
    data_split.train_x = sigmoid.(data_split.train_x)
    data_split.test_x = sigmoid.(data_split.test_x)
end

"""
    FLIRSplit
"""
mutable struct FLIRSplit
    train_x::Matrix{AbstractFloat}
    test_x::Matrix{AbstractFloat}
    video_x::Matrix{AbstractFloat}
    train_y::Vector{Integer}
    test_y::Vector{Integer}
    video_y::Vector{Integer}
end

"""
    FLIRSplit(data_dir::String, extractor::String)
"""
function FLIRSplit(data_dir::String, extractor::String)
    # Get the paths to each h5 file
    train_data = joinpath(data_dir, "train", extractor)
    test_data = joinpath(data_dir, "val", extractor)
    video_data = joinpath(data_dir, "video", extractor)

    # Load x and y from each h5 file
    train_x = h5read(train_data, "df")
    test_x = h5read(test_data, "df")
    video_x = h5read(video_data, "df")
    train_y = h5read(train_data, "classes")
    test_y = h5read(test_data, "classes")
    video_y = h5read(video_data, "classes")

    train_x = convert(Matrix{Float64}, train_x)
    test_x = convert(Matrix{Float64}, test_x)
    video_x = convert(Matrix{Float64}, video_x)
    train_y = convert(Vector{Int64}, train_y)
    test_y = convert(Vector{Int64}, test_y)
    video_y = convert(Vector{Int64}, video_y)

    # Create the FLIR data split
    FLIRSplit(
        train_x,
        test_x,
        video_x,
        train_y,
        test_y,
        video_y
    )
end # FLIRSplit(data_dir::String, extractor::String)

"""
    FLIRL2MSplit
"""
mutable struct FLIRL2MSplit
    train_x::Vector{Matrix{AbstractFloat}}
    test_x::Vector{Matrix{AbstractFloat}}
    video_x::Vector{Matrix{AbstractFloat}}
    train_y::Vector{Vector{Integer}}
    test_y::Vector{Vector{Integer}}
    video_y::Vector{Vector{Integer}}
end


"""
    FLIRL2MSplit(data_dir::String, extractor::String)
"""
function FLIRL2MSplit(data_dir::String, extractor::String)
    # Get the paths to each h5 file
    train_data = joinpath(data_dir, "train", extractor)
    test_data = joinpath(data_dir, "val", extractor)
    video_data = joinpath(data_dir, "video", extractor)

    # Load x and y from each h5 file
    train_x = h5read(train_data, "df")
    test_x = h5read(test_data, "df")
    video_x = h5read(video_data, "df")
    train_y = h5read(train_data, "classes")
    test_y = h5read(test_data, "classes")
    video_y = h5read(video_data, "classes")

    train_x = convert(Matrix{Float64}, train_x)
    test_x = convert(Matrix{Float64}, test_x)
    video_x = convert(Matrix{Float64}, video_x)
    train_y = convert(Vector{Int64}, train_y)
    test_y = convert(Vector{Int64}, test_y)
    video_y = convert(Vector{Int64}, video_y)

    trains_x = []
    tests_x = []
    videos_x = []
    trains_y = []
    tests_y = []
    videos_y = []

    for i = 1:4
        push!(trains_x, train_x[:, train_y .== i])
        push!(tests_x, test_x[:, test_y.== i])
        push!(videos_x, video_x[:, video_y .== i])
        push!(trains_y, train_y[train_y .== i])
        push!(tests_y, test_y[test_y .== i])
        push!(videos_y, video_y[video_y .== i])
    end

    # Create the FLIR data split
    FLIRL2MSplit(
        trains_x,
        tests_x,
        videos_x,
        trains_y,
        tests_y,
        videos_y
    )
end # FLIRL2MSplit(data_dir::String, extractor::String)

"""
FLIRL2MSplitCombined
"""
mutable struct FLIRL2MSplitCombined
    train_x::Vector{Matrix{AbstractFloat}}
    test_x::Vector{Matrix{AbstractFloat}}
    train_y::Vector{Vector{Integer}}
    test_y::Vector{Vector{Integer}}
end # FLIRL2MSplitCombined


"""
    FLIRL2MSplitCombined(data_dir::String, extractor::String)
"""
function FLIRL2MSplitCombined(data_dir::String, extractor::String)
    # Get the paths to each h5 file
    train_data = joinpath(data_dir, "train", extractor)
    test_data = joinpath(data_dir, "val", extractor)
    video_data = joinpath(data_dir, "video", extractor)

    # Load x and y from each h5 file
    train_x = h5read(train_data, "df")
    test_x = h5read(test_data, "df")
    video_x = h5read(video_data, "df")
    train_y = h5read(train_data, "classes")
    test_y = h5read(test_data, "classes")
    video_y = h5read(video_data, "classes")

    train_x = convert(Matrix{Float64}, train_x)
    test_x = convert(Matrix{Float64}, test_x)
    video_x = convert(Matrix{Float64}, video_x)
    train_y = convert(Vector{Int64}, train_y)
    test_y = convert(Vector{Int64}, test_y)
    video_y = convert(Vector{Int64}, video_y)

    trains_x = []
    tests_x = []
    videos_x = []
    trains_y = []
    tests_y = []
    videos_y = []

    for i = 1:4
        push!(trains_x, train_x[:, train_y .== i])
        push!(trains_y, train_y[train_y .== i])

        test_dest_x = test_x[:, test_y.== i]
        test_dest_x = hcat(test_dest_x, video_x[:, video_y .== i])
        push!(tests_x, test_dest_x)

        test_dest_y = test_y[test_y .== i]
        append!(test_dest_y, video_y[video_y .== i])
        push!(tests_y, test_dest_y)
    end

    # Create the FLIR data split
    FLIRL2MSplitCombined(
        trains_x,
        tests_x,
        trains_y,
        tests_y,
    )
end # FLIRL2MSplitCombined(data_dir::String, extractor::String)


# --------------------------------------------------------------------------- #
# FLIR DDVFA SIM
# --------------------------------------------------------------------------- #
"""
    ddvfa_flir_sim(d::Dict{String, Any}, flir_split::FLIRSplit)
"""
function ddvfa_flir_sim(d::Dict{String, Any}, flir_split::FLIRSplit)

    # Set the DDVFA options
    ddvfa_opts = opts_DDVFA()
    ddvfa_opts.method = d["method"]
    ddvfa_opts.gamma = d["gamma"]
    ddvfa_opts.rho_ub = d["rho_ub"]
    ddvfa_opts.rho_lb = d["rho_lb"]
    ddvfa_opts.rho = d["rho_lb"]
    ddvfa_opts.display = false

    # Create the ART modules
    art = DDVFA(ddvfa_opts)

    # Get the data stats
    dim, _ = size(flir_split.train_x)

    # Set the DDVFA config Manually
    art.config = DataConfig(0.0, 1.0, dim)

    # Select which data to test on
    if d["d_te"] == "test"
        test_x = flir_split.test_x
        test_y = flir_split.test_y
    elseif d["d_te"] == "video"
        test_x = flir_split.video_x
        test_y = flir_split.video_y
    else
        error("Incorrect test data mode selected")
    end

    # Logging
    @info "Training: $(d["ext"]), testing: $(d["d_te"])"

    # Train the DDVFA model and time it
    train_stats = @timed train!(art, flir_split.train_x, y=flir_split.train_y)
    y_hat_train = train_stats.value

    # Training performance
    local_train_y = convert(Array{Int}, flir_split.train_y)
    train_perf = NaN
    try
        train_perf = performance(y_hat_train, local_train_y)
    catch
        @info "Performance error!"
    end
    # @info "Training Performance: $(train_perf)"

    # Testing performance, timed
    # test_stats = @timed classify(art, flir_split.test_x, get_bmu=true)
    test_stats = @timed classify(art, test_x, get_bmu=true)
    y_hat_test = test_stats.value
    # local_test_y = convert(Array{Int}, flir_split.test_y)
    local_test_y = convert(Array{Int}, test_y)
    test_perf = NaN
    try
        test_perf = performance(y_hat_test, local_test_y)
    catch
        @info "Performance error!"
    end
    @info "Testing Performance: $(test_perf)"

    # Get the number of weights and categories
    total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
    total_cat = sum(total_vec)
    # @info "Categories: $(art_ddvfa.n_categories)"
    # @info "Weights: $(total_cat)"

    # Store all of the results of interest
    fulld = copy(d)
    # Performances
    fulld["p_tr"] = train_perf
    fulld["p_te"] = test_perf
    # ART statistics
    fulld["n_cat"] = art.n_categories
    fulld["n_wt"] = total_cat
    fulld["m_wt"] = mean(total_vec)
    fulld["s_wt"] = std(total_vec)
    # Timing statistics
    fulld["t_tr"] = train_stats.time
    fulld["gc_tr"] = train_stats.gctime
    fulld["b_tr"] = train_stats.bytes
    fulld["t_te"] = test_stats.time
    fulld["gc_te"] = test_stats.gctime
    fulld["b_te"] = test_stats.bytes
    # Return the results
    return fulld
end # ddvfa_flir_sim(d::Dict{String, Any}, flir_split::FLIRSplit)
