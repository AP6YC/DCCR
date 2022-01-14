using AdaptiveResonance

using StatsBase
using Statistics

using Logging
# using HDF5              # Loading .h5 activation files

using DelimitedFiles

using MLBase

# -------------------------------------------
# Aliases
# -------------------------------------------
#   **Taken from StatsBase.jl**
#
#  These types signficantly reduces the need of using
#  type parameters in functions (which are often just
#  for the purpose of restricting the arrays to real)
#
# These could be removed when the Base supports
# covariant type notation, i.e. AbstractVector{<:Real}

# Real-numbered aliases
const RealArray{T<:Real, N} = AbstractArray{T, N}
const RealVector{T<:Real} = AbstractArray{T, 1}
const RealMatrix{T<:Real} = AbstractArray{T, 2}

# Integered aliases
const IntegerArray{T<:Integer, N} = AbstractArray{T, N}
const IntegerVector{T<:Integer} = AbstractArray{T, 1}
const IntegerMatrix{T<:Integer} = AbstractArray{T, 2}

# Specifically floating-point aliases
const RealFP = Union{Float32, Float64}

# System's largest native floating point variable
const Float = (Sys.WORD_SIZE == 64 ? Float64 : Float32)

# """
#     DataSplit

# A basic struct for encapsulating the four components of supervised training.
# """
# mutable struct DataSplit
#     train_x::Array
#     test_x::Array
#     train_y::Array
#     test_y::Array
#     DataSplit(train_x, test_x, train_y, test_y) = new(train_x, test_x, train_y, test_y)
# end

# """
#     DataSplit(data_x::Array, data_y::Array, ratio::Float)

# Return a DataSplit struct that is split by the ratio (e.g. 0.8).
# """
# function DataSplit(data_x::Array, data_y::Array, ratio::Real)
#     dim, n_data = size(data_x)
#     split_ind = Int(floor(n_data * ratio))

#     train_x = data_x[:, 1:split_ind]
#     test_x = data_x[:, split_ind + 1:end]
#     train_y = data_y[1:split_ind]
#     test_y = data_y[split_ind + 1:end]

#     return DataSplit(train_x, test_x, train_y, test_y)
# end

# """
#     DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

# Sequential loading and ratio split of the data.
# """
# function DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)
#     dim, n_data = size(data_x)
#     n_splits = length(seq_ind)

#     train_x = Array{Float64}(undef, dim, 0)
#     train_y = Array{Float64}(undef, 0)
#     test_x = Array{Float64}(undef, dim, 0)
#     test_y = Array{Float64}(undef, 0)

#     # Iterate over all splits
#     for ind in seq_ind
#         local_x = data_x[:, ind[1]:ind[2]]
#         local_y = data_y[ind[1]:ind[2]]
#         # n_data = ind[2] - ind[1] + 1
#         n_data = size(local_x)[2]
#         split_ind = Int(floor(n_data * ratio))

#         train_x = [train_x local_x[:, 1:split_ind]]
#         test_x = [test_x local_x[:, split_ind + 1:end]]
#         train_y = [train_y; local_y[1:split_ind]]
#         test_y = [test_y; local_y[split_ind + 1:end]]
#     end
#     return DataSplit(train_x, test_x, train_y, test_y)
# end # DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)


"""
    sigmoid(x::Real)

Return the sigmoid function on x.
"""
function sigmoid(x::Real)
# return 1.0 / (1.0 + exp(-x))
    return one(x) / (one(x) + exp(-x))
end

"""
    collect_activations(data_dir::String)

Return the activations from a single directory
"""
function collect_activations(data_dir::String)
    data_full = readdlm(joinpath(data_dir, "average_features.csv"), ',')
    return data_full
end

"""
    collect_all_activations(data_dirs::Array, cell::Int)

Return just the yolo activations from a list of data directories.
"""
function collect_all_activations(data_dirs::Array, cell::Int)
    data_grand = []
    for data_dir in data_dirs
        data_dir_full = joinpath(data_dir, string(cell))
        data_full = collect_activations(data_dir_full)
        # If the full data struct is empty, initialize with the size of the data
        if isempty(data_grand)
            data_grand = Array{Float64}(undef, size(data_full)[1], 1)
        end
        data_grand = [data_grand data_full]
    end
    return data_grand
end

"""
    collect_all_activations_labeled(data_dirs::Vector{String}, cell::Int)

Return the yolo activations, training targets, and condensed labels list from a list of data directories.
"""
function collect_all_activations_labeled(data_dirs::Vector{String}, cell::Int)
    top_dim = 128*cell
    data_grand = Matrix{Float64}(undef, top_dim, 0)
    targets = Vector{Int64}()
    labels = Vector{String}()
    # for data_dir in data_dirs
    for i = 1:length(data_dirs)
        # Get the full local data directory
        data_dir = data_dirs[i]
        data_dir_full = joinpath(data_dir, string(cell))

        # Assign the directory as the label
        push!(labels, basename(data_dir))

        # Get all of the data from the full data directory
        data_full = collect_activations(data_dir_full)
        dim, n_samples = size(data_full)

        # If the full data struct is empty, initialize with the size of the data
        if isempty(data_grand)
            data_grand = Array{Float64}(undef, dim, 0)
        end

        # Set the labeled targets
        for j = 1:n_samples
            push!(targets, i)
        end

        # Concatenate the most recent batch with the grand dataset
        data_grand = [data_grand data_full]
    end
    return data_grand, targets, labels
end

"""
    get_dist(data::RealMatrix)

Get the distribution parameters for preprocessing.
"""
function get_dist(data::RealMatrix)
    return fit(ZScoreTransform, data, dims=2)
end

"""
    function_preprocess(dt::ZScoreTransform, scaling::Real, data::RealMatrix)

Preprocesses one dataset of features, scaling and squashing along the feature axes.
"""
function feature_preprocess(dt::ZScoreTransform, scaling::Real, data::RealMatrix)
    new_data = StatsBase.transform(dt, data)
    new_data = sigmoid.(scaling*new_data)
    return new_data
end

"""
    DataSplit

A basic struct for encapsulating the components of supervised training.
"""
mutable struct DataSplit

    train_x::RealMatrix
    train_y::IntegerVector
    train_labels::Vector{String}

    val_x::RealMatrix
    val_y::IntegerVector
    val_labels::Vector{String}

    test_x::RealMatrix
    test_y::RealVector
    test_labels::Vector{String}

    DataSplit(
        train_x,
        train_y,
        train_labels,
        val_x,
        val_y,
        val_labels,
        test_x,
        test_y,
        test_labels
    ) = new(
        train_x,
        train_y,
        train_labels,
        val_x,
        val_y,
        val_labels,
        test_x,
        test_y,
        test_labels
    )
end

"""
    DataSplitIndexed

A basic struct for encapsulating the components of supervised training.
"""
mutable struct DataSplitIndexed
    train_x::Vector{RealMatrix}
    train_y::Vector{IntegerVector}
    train_labels::Vector{String}

    val_x::Vector{RealMatrix}
    val_y::Vector{IntegerVector}
    val_labels::Vector{String}

    test_x::Vector{RealMatrix}
    test_y::Vector{IntegerVector}
    test_labels::Vector{String}
end

"""
    load_orbits(data_dir::String, scaling::Real)

Load the orbits data and preprocess the features.
"""
function load_orbits(data_dir::String, scaling::Real)
    train_dir = joinpath(data_dir, "LBs")
    val_dir = joinpath(data_dir, "Val")
    test_dir = joinpath(data_dir, "EBs")

    train_data_dirs = [joinpath(train_dir, data_dir) for data_dir in data_dirs]
    val_data_dirs = [joinpath(val_dir, data_dir) for data_dir in data_dirs]
    test_data_dirs = [joinpath(test_dir, data_dir) for data_dir in data_dirs]

    train_x, train_y, train_labels = collect_all_activations_labeled(train_data_dirs, 1)
    val_x, val_y, val_labels = collect_all_activations_labeled(train_data_dirs, 1)
    test_x, test_y, test_labels = collect_all_activations_labeled(test_data_dirs, 1)

    dt = get_dist(train_x)

    train_x = feature_preprocess(dt, scaling, train_x)
    test_x = feature_preprocess(dt, scaling, test_x)

    data_struct = DataSplit(
        train_x,
        train_y,
        train_labels,
        val_x,
        val_y,
        val_labels,
        test_x,
        test_y,
        test_labels
    )

    return data_struct
    # return X_train, y_train, train_labels, X_test, y_test, test_labels
end

function get_indexed_data(data::DataSplit)
    # Assume the same number of classes in each category
    n_classes = length(unique(data.train_y))

    # data_indexed =
    train_x = Vector{RealMatrix}()
    train_y = Vector{IntegerVector}()
    train_labels = Vector{String}()
    val_x = Vector{RealMatrix}()
    val_y = Vector{IntegerVector}()
    val_labels = Vector{String}()
    test_x = Vector{RealMatrix}()
    test_y = Vector{IntegerVector}()
    test_labels = Vector{String}()

    for i = 1:n_classes
        i_train = findall(x -> x == i, data.train_y)
        push!(train_x, data.train_x[:, i_train])
        push!(train_y, data.train_y[i_train])
        i_val = findall(x -> x == i, data.val_y)
        push!(val_x, data.val_x[:, i_val])
        push!(val_y, data.val_y[i_val])
        i_test = findall(x -> x == i, data.test_y)
        push!(test_x, data.test_x[:, i_test])
        push!(test_y, data.test_y[i_test])
    end

    train_labels = data.train_labels
    val_labels = data.val_labels
    test_labels = data.test_labels

    # println(typeof(train_x))
    # println(typeof(train_y))
    # println(typeof(train_labels))

    data_indexed = DataSplitIndexed(
        train_x,
        train_y,
        train_labels,
        val_x,
        val_y,
        val_labels,
        test_x,
        test_y,
        test_labels
    )
    return data_indexed
end

function get_orbit_names(selection::Vector{String})
    # Data directories to train/test on
    data_dirs = Dict(
        "dot_dusk" => "dot_dusk",
        "dot_morning" => "dot_morning",
        "emahigh_dusk" => "emahigh_dusk",
        "emahigh_morning" => "emahigh_morning",
        "emalow_dusk" => "emalow_dusk",
        "emalow_morning" => "emalow_morning",
        "pr_dusk" => "pr_dusk",
        "pr_morning" => "pr_morning",
    )

    class_labels = Dict(
        "dot_dusk" => "DOTD",
        "dot_morning" => "DOTM",
        "emahigh_dusk" => "EMAHD",
        "emahigh_morning" => "EMAHM",
        "emalow_dusk" => "EMALD",
        "emalow_morning" => "EMALM",
        "pr_dusk" => "PRD",
        "pr_morning" => "PRM",
    )

    out_data_dirs = String[]
    out_class_labels = String[]
    for item in selection
        push!(out_data_dirs, data_dirs[item])
        push!(out_class_labels, class_labels[item])
    end

    return out_data_dirs, out_class_labels
end

"""
    get_confusion(n_classes, y, y_hat)

Wrapper method for getting the raw confusion matrix.
"""
function get_confusion(n_classes, y, y_hat)
    return confusmat(n_classes, y, y_hat)
end

"""
    get_normalized_confusion(n_classes, y, y_hat)

Get the normalized confusion matrix.
"""
function get_normalized_confusion(n_classes, y, y_hat)
    cm = get_confusion(n_classes, y, y_hat)
    total = sum(cm, dims=1)
    norm_cm = cm./total
    return norm_cm
end

"""
    get_accuracies(y, y_hat)

Get a list of the percentage accuracies.
"""
function get_accuracies(y, y_hat, n_classes)
    cm = get_confusion(n_classes, y, y_hat)
    correct = [cm[i,i] for i = 1:n_classes]
    total = sum(cm, dims=1)
    accuracies = correct'./total

    return accuracies
end

"""
    get_tt_accuracies(data, y_hat, y_hat)

Get two lists of the training and testing accuracies
"""
function get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    # TRAIN: Get the percent correct for each class
    train_accuracies = get_accuracies(data.train_y, y_hat_train, n_classes)

    # TEST: Get the percent correct for each class
    test_accuracies = get_accuracies(data.test_y, y_hat, n_classes)

    return train_accuracies, test_accuracies
end

function get_n_categories(ddvfa)
    # Save the number of F2 nodes and total categories per class
    n_F2 = Int[]
    n_categories = Int[]

    # Iterate over every class
    for i = 1:n_classes
        # Find all of the F2 nodes that correspond to the class
        i_F2 = findall(x->x==i, ddvfa.labels)
        # Add the number of F2 nodes to the list
        push!(n_F2, length(i_F2))
        # Get the numbers of categories within each F2 node
        n_cat_list = [F2.n_categories for F2 in ddvfa.F2[i_F2]]
        # Sum those and add them to the list
        push!(n_categories, sum(n_cat_list))
    end

    return n_F2, n_categories
end

function create_confusion_heatmap(class_labels, y, y_hat)
    # Number of classes from the class labels
    n_classes = length(class_labels)

    # Normalized confusion
    norm_cm = get_normalized_confusion(n_classes, y, y_hat)

    # Create the heatmap
    h = heatmap(
        class_labels,
        class_labels,
        norm_cm,
        fill_z = norm_cm,
        aspect_ratio=:equal
    )

    # Create the annotations
    fontsize = 15
    nrow, ncol = size(norm_cm)
    ann = [
        (i-.5,j-.5, text(round(norm_cm[i,j], digits=2), fontsize, :white, :center))
        for i in 1:nrow for j in 1:ncol
    ]
    annotate!(ann, linecolor=:white)

    return h
end

"""
    create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels)

Return a grouped bar chart with class accuracies.
"""
function create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels)
    # Infer the number of classes from the class labels
    n_classes = length(class_labels)
    # Get the training and testing accuracies
    train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    @info "Train Accuracies:" train_accuracies
    @info "Train Accuracies:" test_accuracies

    # Format the accuracy series for plotting
    combined_accuracies = [train_accuracies; test_accuracies]'

    # Create the accuracy grouped bar chart
    p = groupedbar(
        combined_accuracies,
        bar_position = :dodge,
        bar_width=0.7,
        dpi=dpi,
        # show=true,
        # xticks=train_labels
    )

    ylabel!(p, "Accuracy")
    xticks!(collect(1:n_classes), class_labels)
    # title!(p, "test")

    return p
end

# --------------------------------------------------------------------------- #
# Linear example
# --------------------------------------------------------------------------- #
# function accuracy_sim(d::Dict{String, Any}, data_split::DataSplit)

#     # Set the DDVFA options
#     ddvfa_opts = opts_DDVFA()
#     ddvfa_opts.method = d["method"]
#     ddvfa_opts.gamma = d["gamma"]
#     ddvfa_opts.rho_ub = d["rho_ub"]
#     ddvfa_opts.rho_lb = d["rho_lb"]
#     # ddvfa_opts.rho = d["rho"]
#     ddvfa_opts.rho = d["rho_lb"]
#     ddvfa_opts.display = false

#     # Create the ART modules
#     art_ddvfa = DDVFA(ddvfa_opts)

#     # Get the data stats
#     dim, n_samples = size(data_split.train_x)

#     # Set the DDVFA config Manually
#     art_ddvfa.config = DataConfig(0.0, 1.0, dim)
#     @info "dim: $dim"

#     # Train the DDVFA model and time it
#     train_stats = @timed train!(art_ddvfa, data_split.train_x, y=data_split.train_y)
#     y_hat_train = train_stats.value

#     # Training performance
#     local_train_y = convert(Array{Int}, data_split.train_y)
#     train_perf = NaN
#     try
#         train_perf = performance(y_hat_train, local_train_y)
#     catch
#         @info "Performance error!"
#     end
#     @info "Training Performance: $(train_perf)"

#     # Testing performance, timed
#     test_stats = @timed classify(art_ddvfa, data_split.test_x)
#     y_hat_test = test_stats.value
#     local_test_y = convert(Array{Int}, data_split.test_y)
#     test_perf = NaN
#     try
#         test_perf = performance(y_hat_test, local_test_y)
#     catch
#         @info "Performance error!"
#     end
#     @info "Testing Performance: $(test_perf)"

#     # Get the number of weights and categories
#     total_vec = [art_ddvfa.F2[i].n_categories for i = 1:art_ddvfa.n_categories]
#     total_cat = sum(total_vec)
#     # @info "Categories: $(art_ddvfa.n_categories)"
#     # @info "Weights: $(total_cat)"

#     # Store all of the results of interest
#     fulld = copy(d)
#     # Performances
#     fulld["p_tr"] = train_perf
#     fulld["p_te"] = test_perf
#     # ART statistics
#     fulld["n_cat"] = art_ddvfa.n_categories
#     fulld["n_wt"] = total_cat
#     fulld["m_wt"] = mean(total_vec)
#     fulld["s_wt"] = std(total_vec)
#     # Timing statistics
#     fulld["t_tr"] = train_stats.time
#     fulld["gc_tr"] = train_stats.gctime
#     fulld["b_tr"] = train_stats.bytes
#     fulld["t_te"] = test_stats.time
#     fulld["gc_te"] = test_stats.gctime
#     fulld["b_te"] = test_stats.bytes
#     # Return the results
#     return fulld
# end
