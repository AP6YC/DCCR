using AdaptiveResonance

using StatsBase
# using Statistics

using Logging
# using HDF5              # Loading .h5 activation files

using DelimitedFiles

using MLBase        # confusmat
# using DrWatson
using MLDataUtils   # stratifiedobs
using StatsPlots    # groupedbar
using DataFrames
using Printf

# Add the custom colors definitions
include("colors.jl")

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

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

    # DataSplit(
    #     train_x,
    #     train_y,
    #     train_labels,
    #     val_x,
    #     val_y,
    #     val_labels,
    #     test_x,
    #     test_y,
    #     test_labels
    # ) = new(
    #     train_x,
    #     train_y,
    #     train_labels,
    #     val_x,
    #     val_y,
    #     val_labels,
    #     test_x,
    #     test_y,
    #     test_labels
    # )
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
    DataSplitCombined

A struct for combining training and validation data, containing only train and test splits.
"""
mutable struct DataSplitCombined
    train_x::RealMatrix
    train_y::IntegerVector
    train_labels::Vector{String}

    test_x::RealMatrix
    test_y::IntegerVector
    test_labels::Vector{String}
end

function DataSplitCombined(data::DataSplit)
    # println(size(data.train_x))
    # println(size(data.val_x))
    DataSplitCombined(
        hcat(data.train_x, data.val_x),
        vcat(data.train_y, data.val_y),
        vcat(data.train_labels, data.val_labels),
        data.test_x,
        data.test_y,
        data.test_labels
    )
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
    val_x, val_y, val_labels = collect_all_activations_labeled(val_data_dirs, 1)
    test_x, test_y, test_labels = collect_all_activations_labeled(test_data_dirs, 1)

    dt = get_dist(train_x)

    train_x = feature_preprocess(dt, scaling, train_x)
    val_x = feature_preprocess(dt, scaling, val_x)
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

# """
#     load_val_orbits(data_dir::String, scaling::Real)

# Load the orbits data and preprocess the features.

# This uses training data for validation because of a problem with the val
# distribution at the time of this writing.
# """
# function load_val_orbits(data_dir::String, scaling::Real)
#     train_dir = joinpath(data_dir, "LBs")
#     # val_dir = joinpath(data_dir, "Val")
#     test_dir = joinpath(data_dir, "EBs")

#     train_data_dirs = [joinpath(train_dir, data_dir) for data_dir in data_dirs]
#     # val_data_dirs = [joinpath(val_dir, data_dir) for data_dir in data_dirs]
#     test_data_dirs = [joinpath(test_dir, data_dir) for data_dir in data_dirs]

#     train_x, train_y, train_labels = collect_all_activations_labeled(train_data_dirs, 1)
#     # val_x, val_y, val_labels = collect_all_activations_labeled(val_data_dirs, 1)
#     test_x, test_y, test_labels = collect_all_activations_labeled(test_data_dirs, 1)

#     dt = get_dist(train_x)

#     train_x = feature_preprocess(dt, scaling, train_x)
#     test_x = feature_preprocess(dt, scaling, test_x)

#     data_struct = DataSplit(
#         train_x,
#         train_y,
#         train_labels,
#         val_x,
#         val_y,
#         val_labels,
#         test_x,
#         test_y,
#         test_labels
#     )

#     return data_struct
#     # return X_train, y_train, train_labels, X_test, y_test, test_labels
# end

"""
    get_indexed_data(data::DataSplit)

Create a DataSplitIndexed object from a DataSplit.
"""
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

    # Construct the indexed data split
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

"""
    get_deindexed_data(data::DataSplitIndexed, order::IntegerVector)

Turn a DataSplitIndexed into a DataSplit with the given train/test order.
"""
function get_deindexed_data(data::DataSplitIndexed, order::IntegerVector)
    dim = 128
    train_x = Array{Float64}(undef, dim, 0)
    train_y = Array{Int}(undef, 0)
    train_labels = Vector{String}()

    val_x = Array{Float64}(undef, 128, 0)
    val_y = Array{Int}(undef, 0)
    val_labels = Vector{String}()

    test_x = Array{Float64}(undef, 128, 0)
    test_y = Array{Int}(undef, 0)
    test_labels = Vector{String}()

    for i in order
        train_x = hcat(train_x, data.train_x[i])
        train_y = vcat(train_y, data.train_y[i])
        val_x = hcat(val_x, data.val_x[i])
        val_y = vcat(val_y, data.val_y[i])
        test_x = hcat(test_x, data.test_x[i])
        test_y = vcat(test_y, data.test_y[i])
    end

    train_labels = data.train_labels[order]
    val_labels = data.val_labels[order]
    test_labels = data.test_labels[order]

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
end

"""
    get_orbit_names(selection::Vector{String})

Map the experiment orbit names to their data directories and plotting class labels.
"""
function get_orbit_names(selection::Vector{String})
    # Data directory names
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

    # Class labels for plotting
    class_labels = Dict(
        "dot_dusk" => "DOTD",
        "dot_morning" => "DOTM",
        "emahigh_dusk" => "EMAHD",
        "emahigh_morning" => "EMAHM",
        # "emalow_dusk" => "EMALD",
        # "emalow_morning" => "EMALM",
        "emalow_dusk" => "EMAD",
        "emalow_morning" => "EMAM",
        "pr_dusk" => "PRD",
        "pr_morning" => "PRM",
    )

    # Create the output lists
    out_data_dirs = String[]
    out_class_labels = String[]
    for item in selection
        push!(out_data_dirs, data_dirs[item])
        push!(out_class_labels, class_labels[item])
    end

    return out_data_dirs, out_class_labels
end

"""
    get_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Int)

Wrapper method for getting the raw confusion matrix.
"""
function get_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Int)
    return confusmat(n_classes, y, y_hat)
end

"""
    get_normalized_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Int)

Get the normalized confusion matrix.
"""
function get_normalized_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Int)
    cm = get_confusion(y, y_hat, n_classes)
    # total = sum(cm, dims=1)
    total = sum(cm, dims=2)'
    norm_cm = cm./total
    return norm_cm
end

"""
    get_accuracies(y::IntegerVector, y_hat::IntegerVector, n_classes::Int)

Get a list of the percentage accuracies.
"""
function get_accuracies(y::IntegerVector, y_hat::IntegerVector, n_classes::Int)
    cm = get_confusion(y, y_hat, n_classes)
    correct = [cm[i,i] for i = 1:n_classes]
    # total = sum(cm, dims=1)
    total = sum(cm, dims=2)'
    accuracies = correct'./total

    return accuracies
end

"""
    get_tt_accuracies(data::Union{DataSplit, DataSplitCombined}, y_hat_train::IntegerVector, y_hat::IntegerVector, n_classes::Int)

Get two lists of the training and testing accuracies
"""
function get_tt_accuracies(data::Union{DataSplit, DataSplitCombined}, y_hat_train::IntegerVector, y_hat::IntegerVector, n_classes::Int)
    # TRAIN: Get the percent correct for each class
    train_accuracies = get_accuracies(data.train_y, y_hat_train, n_classes)

    # TEST: Get the percent correct for each class
    test_accuracies = get_accuracies(data.test_y, y_hat, n_classes)

    return train_accuracies, test_accuracies
end

"""
    get_n_categories(ddvfa::DDVFA)

Returns both the number of F2 categories and total number of weights per class as two lists.
"""
function get_n_categories(ddvfa::DDVFA)
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

"""
    get_manual_split(data::RealMatrix, targets::IntegerVector)

Wrapper, returns a manual train/test x/y split from a data matrix and labels using MLDataUtils.
"""
function get_manual_split(data::RealMatrix, targets::IntegerVector)
    (X_train, y_train), (X_test, y_test) = stratifiedobs((data, targets))
    return (X_train, y_train), (X_test, y_test)
end

"""
    df_column_to_matrix(df::DataFrame, row::Symbol)

Convert a column of lists in a DataFrame into a matrix for analysis.
"""
function df_column_to_matrix(df::DataFrame, row::Symbol)
    lists = df[!, row]
    n_samples = length(lists)
    n_classes = length(lists[1])
    matrix = zeros(n_samples, n_classes)
    for i = 1:n_samples
        matrix[i, :] = lists[i]
    end
    return matrix
end

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# """
#     create_confusion_heatmap_old(class_labels::Vector{String}, y::IntegerVector, y_hat::IntegerVector)

# Returns a handle to a labeled and annotated heatmap plot of the confusion matrix.
# """
# function create_confusion_heatmap_old(class_labels::Vector{String}, y::IntegerVector, y_hat::IntegerVector)
#     # Number of classes from the class labels
#     n_classes = length(class_labels)

#     # Normalized confusion
#     norm_cm = get_normalized_confusion(y, y_hat, n_classes)

#     # Transpose reflect
#     plot_cm = norm_cm'

#     # Create the heatmap
#     h = heatmap(
#         class_labels,
#         class_labels,
#         plot_cm,
#         fill_z = norm_cm,
#         aspect_ratio=:equal,
#         color = cgrad(GRADIENTSCHEME),
#         fontfamily=FONTFAMILY,
#         annotationfontfamily=FONTFAMILY,
#         size=SQUARE_SIZE,
#         dpi=DPI
#     )

#     # Create the annotations
#     fontsize = 10
#     nrow, ncol = size(norm_cm)
#     ann = [
#         (
#             i-.5,
#             j-.5,
#             text(
#                 round(plot_cm[i,j], digits=2),
#                 fontsize,
#                 FONTFAMILY,
#                 :white,
#                 :center,
#             )
#         )
#         for i in 1:nrow for j in 1:ncol
#     ]

#     # Add the cell annotations
#     annotate!(
#         ann,
#         linecolor=:white,
#         # linecolor=:black,
#         fontfamily=FONTFAMILY,
#     )

#     # Label truth and predicted axes
#     xlabel!("Predicted")
#     ylabel!("Truth")

#     # Return the plot handle for display or saving
#     return h
# end

"""
    create_confusion_heatmap(class_labels::Vector{String}, y::IntegerVector, y_hat::IntegerVector)

Returns a handle to a labeled and annotated heatmap plot of the confusion matrix.
"""
function create_confusion_heatmap(class_labels::Vector{String}, y::IntegerVector, y_hat::IntegerVector)
    # Number of classes from the class labels
    n_classes = length(class_labels)
    # Normalized confusion
    norm_cm = get_normalized_confusion(y, y_hat, n_classes)
    # Transpose reflect
    # plot_cm = reverse(norm_cm', dims=1)
    plot_cm = reverse(norm_cm, dims=1)
    # Convert to percentages
    plot_cm *= 100.0
    # Transpose the y labels
    x_labels = class_labels
    y_labels = reverse(class_labels)

    # Create the heatmap
    h = heatmap(
        x_labels,
        y_labels,
        plot_cm,
        fill_z = norm_cm,
        aspect_ratio=:equal,
        color = cgrad(GRADIENTSCHEME),
        clims = (0, 100),
        fontfamily=FONTFAMILY,
        annotationfontfamily=FONTFAMILY,
        size=SQUARE_SIZE,
        dpi=DPI
    )

    # Create the annotations
    fontsize = 10
    nrow, ncol = size(norm_cm)
    ann = [
        (
            i-.5,
            j-.5,
            text(
                round(plot_cm[j,i], digits=2),
                fontsize,
                FONTFAMILY,
                :white,
                :center,
            )
        )
        for i in 1:nrow for j in 1:ncol
    ]

    # Add the cell annotations
    annotate!(
        ann,
        linecolor=:white,
        # linecolor=:black,
        fontfamily=FONTFAMILY,
    )

    plot!(
        bottom_margin = -8Plots.mm,
    )

    # Label truth and predicted axes
    xlabel!("Predicted")
    ylabel!("Truth")

    # Return the plot handle for display or saving
    return h
end

"""
    create_custom_confusion_heatmap(class_labels::Vector{String}, y::IntegerVector, y_hat::IntegerVector)

Returns a handle to a labeled and annotated heatmap plot of the confusion matrix.
"""
function create_custom_confusion_heatmap(class_labels::Vector{String}, norm_cm)
    # Number of classes from the class labels
    # n_classes = length(class_labels)
    # Normalized confusion
    # norm_cm = get_normalized_confusion(y, y_hat, n_classes)
    # Transpose reflect
    plot_cm = reverse(norm_cm', dims=1)
    # Convert to percentages
    plot_cm *= 100.0
    # Transpose the y labels
    x_labels = class_labels
    y_labels = reverse(class_labels)

    # Create the heatmap
    h = heatmap(
        x_labels,
        y_labels,
        plot_cm,
        fill_z = norm_cm,
        aspect_ratio=:equal,
        color = cgrad(GRADIENTSCHEME),
        clims = (0, 100),
        fontfamily=FONTFAMILY,
        annotationfontfamily=FONTFAMILY,
        size=SQUARE_SIZE,
        dpi=DPI
    )

    # Create the annotations
    fontsize = 10
    nrow, ncol = size(norm_cm)
    ann = [
        (
            i-.5,
            j-.5,
            text(
                # round(plot_cm[j,i], digits=2),
                string(round(plot_cm[j,i], digits=2)) * "%",
                fontsize,
                FONTFAMILY,
                :white,
                :center,
            )
        )
        for i in 1:nrow for j in 1:ncol
    ]

    # Add the cell annotations
    annotate!(
        ann,
        linecolor=:white,
        # linecolor=:black,
        fontfamily=FONTFAMILY,
    )

    plot!(
        bottom_margin = -8Plots.mm,
    )

    # Label truth and predicted axes
    xlabel!("Predicted")
    ylabel!("Truth")

    # Return the plot handle for display or saving
    return h
end

"""
    create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels)

Return a grouped bar chart with class accuracies.
"""
function create_accuracy_groupedbar(data, y_hat_train, y_hat, class_labels ; percentages=false)
    # Infer the number of classes from the class labels
    n_classes = length(class_labels)

    # Get the training and testing accuracies
    train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    @info "Train Accuracies:" train_accuracies
    @info "Train Accuracies:" test_accuracies

    # Format the accuracy series for plotting
    combined_accuracies = [train_accuracies; test_accuracies]'

    # Convert to percentages
    y_formatter = percentages ? percentage_formatter : :auto

    # Create the accuracy grouped bar chart
    p = groupedbar(
        combined_accuracies,
        bar_position = :dodge,
        bar_width=0.7,
        color_palette=COLORSCHEME,
        fontfamily=FONTFAMILY,
        legend_position=:outerright,
        labels=["Training" "Testing"],
        dpi=DPI,
        yformatter = y_formatter,
        # yformatter = j -> @sprintf("%0.0f%%", 100*j),
        # show=true,
        # xticks=train_labels
    )

    ylabel!(p, "Context Accuracy")
    # yticklabels(j -> @sprintf("%0.0f%%", 100*j))
    xticks!(collect(1:n_classes), class_labels)
    # title!(p, "test")

    return p
end

"""
    create_comparison_groupedbar(data, y_hat_val, y_hat, class_labels)

Return a grouped bar chart with comparison bars.
"""
function create_comparison_groupedbar(data, y_hat_val, y_hat, class_labels ; percentages=false, extended=false)
    # Infer the number of classes from the class labels
    n_classes = length(class_labels)
    # If we need a category to dump in, extend the
    # extended && n_classes+= 1
    if extended
        n_classes += 1
    end

    # Get the training and testing accuracies
    # train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    test_accuracies = get_accuracies(data.test_y, y_hat, n_classes)
    test_accuracies_val = get_accuracies(data.test_y, y_hat_val, n_classes)

    # If we had a dumping category, pop the last results to get back
    @info test_accuracies
    # if extended
        # test_accuracies = test_accuracies[1, 1:end-1]
        # test_accuracies_val = test_accuracies_val[1, 1:end-1]
    # end

    # Format the accuracy series for plotting
    combined_accuracies = [test_accuracies; test_accuracies_val]'

    # Convert to percentages
    y_formatter = percentages ? percentage_formatter : :auto

    # Create the accuracy grouped bar chart
    p = groupedbar(
        combined_accuracies,
        bar_position = :dodge,
        bar_width=0.7,
        color_palette=COLORSCHEME,
        fontfamily=FONTFAMILY,
        legend_position=:outerright,
        labels=["Before" "After"],
        dpi=DPI,
        yformatter = y_formatter,
        # yformatter = j -> @sprintf("%0.0f%%", 100*j),
        # show=true,
        # xticks=train_labels
    )

    ylabel!(p, "Context Accuracy")
    # yticklabels(j -> @sprintf("%0.0f%%", 100*j))
    if extended
        xticks!(collect(1:n_classes-1), class_labels)
    else
        xticks!(collect(1:n_classes), class_labels)
    end
    xticks!(collect(1:n_classes), class_labels)
    # title!(p, "test")

    return p
end

"""
    create_boxplot(data::RealMatrix, class_labels::Vector{String})

Return a colored and formatted boxplot of the data.
"""
function create_boxplot(data::RealMatrix, class_labels::Vector{String} ; percentages=false)
    # Get the number of sample vectors
    # n_samples = size(n_w_matrix)[1]
    n_samples = size(data)[1]
    # Vectorize the data along the columns
    new_matrix = vec(data)
    # Convert to percentages
    y_formatter = percentages ? percentage_formatter : :auto
    # Label each sample with an inner-repeated label list
    new_labels = repeat(class_labels, inner=n_samples)
    # Create a dataframe with each sample and class label
    df = DataFrame([new_matrix, new_labels], ["n_w", "class"])

    # Create a violin plot
    p = @df df violin(
        :class,
        :n_w,
        linewidth=0,
        color_palette=COLORSCHEME,
        fontfamily=FONTFAMILY,
    )

    # Overlay a transparent box plot
    @df df boxplot!(
        :class,
        :n_w,
        fillalpha=0.75,
        linewidth=2,
        color_palette=COLORSCHEME,
        # fontfamily=FONTFAMILY,
    )

    if percentages
        # ylims!(p, (-Inf, 1))
        # ylims!(p, (0.6, 1))
        ylims!(p, PERCENTAGES_BOUNDS)
    end

    # Format the plot
    plot!(
        dpi=DPI,
        legend=false,
        yformatter=y_formatter,
        # color_palette=COLORSCHEME,
    )

    # Add the universal x-label
    xlabel!("Context")

    return p
end

"""
    create_inverted_boxplot(data::RealMatrix, class_labels::Vector{String})

Return a colored and formatted boxplot of the data.
"""
function create_inverted_boxplot(data::RealMatrix, class_labels::Vector{String} ; percentages=false)
    # Get the number of sample vectors
    n_samples = size(data)[1]
    # Vectorize the data along the columns
    new_matrix = vec(data)
    # Convert to percentages
    y_formatter = percentages ? percentage_formatter : :auto
    # Label each sample with an inner-repeated label list
    new_labels = repeat(class_labels, inner=n_samples)
    # Create a dataframe with each sample and class label
    df = DataFrame([new_matrix, new_labels], ["n_w", "class"])

    local_palette = palette(COLORSCHEME)

    # p = @df df dotplot(
    #     :class,
    #     :n_w,
    # )

    # Overlay a transparent box plot
    p = @df df boxplot(
    # @df df boxplot!(
        :class,
        :n_w,
        fillalpha=0.75,
        linewidth=2,
        color=local_palette[2],
        order=2,
        fontfamily=FONTFAMILY,
    )

    # Create a violin plot
    @df df violin!(
        :class,
        :n_w,
        linewidth=0,
        fillalpha=0.75,
        color=local_palette[1],
        # fontfamily=FONTFAMILY,
        order=1,
    )

    if percentages
        # ylims!(p, (-Inf, 1))
        # ylims!(p, (0.6, 1))
        ylims!(p, PERCENTAGES_BOUNDS)
    end

    # Format the plot
    plot!(
        dpi=DPI,
        legend=false,
        yformatter=y_formatter,
    )

    # Add the universal x-label
    xlabel!("Context")

    return p
    # return p, df
end

"""
    create_condensed_plot(y_hat, class_labels)

Create and return a simplified condensed scenario plot.
"""
function create_condensed_plot(perfs, class_labels, percentages=true)
    # Add initial testing block to labels
    local_labels = cat("", class_labels, dims=1)
    println(local_labels)
    # local_labels = reshape(local_labels, 1, length(local_labels))

    # Convert to percentages
    # plot_perfs = perfs * 100.0;
    y_formatter = percentages ? percentage_formatter : :auto

    p = plot(
        # plot_perfs,
        perfs,
        linestyle = [:dot :dash :dashdot :solid :dot :dashdotdot],
        # linestyle = :auto,
        linewidth = 3,
        # thickness_scaling = 1,
        color_palette=COLORSCHEME,
        labels=reshape(class_labels, 1, length(class_labels)),
        # legend=:topleft,
        fontfamily=FONTFAMILY,
        legend=:outerright,
        yformatter=y_formatter,
        # legendlinewidth=10,
        dpi=DPI,
    )

    xlabel!("Training Context")
    ylabel!("Testing Accuracy")
    xticks!(collect(1:length(local_labels)), local_labels)

    return p
end

"""
    create_complex_condensed_plot(y_hat, vals, class_labels)

Create and return a complex condensed scenario plot.
"""
function create_complex_condensed_plot(perfs, vals, class_labels, percentages=true)
    # Add initial testing block to labels
    local_labels = cat("", class_labels, dims=1)
    println(local_labels)
    # local_labels = reshape(local_labels, 1, length(local_labels))

    # Convert to percentages
    # plot_perfs = perfs * 100.0;
    y_formatter = percentages ? percentage_formatter : :auto

    linestyles = [:dot :dash :dashdot :solid :dot :dashdotdot]
    linewidths = 2

    n_classes = length(class_labels)
    plot_data = Array{Float64}(undef, n_classes, 0)

    n_eb = N_EB

    cutoffs = []

    # First EB
    local_eb = [perfs[j][1] for j = 1:n_classes]
    local_eb = repeat(local_eb, outer=[1, n_eb])
    plot_data = hcat(plot_data, local_eb)
    push!(cutoffs, n_eb)

    for i = 1:n_classes
        # Append the validation values
        plot_data = hcat(plot_data, vals[i])
        push!(cutoffs, cutoffs[end] + size(vals[i], 2))
        # Create an EB
        local_eb = [perfs[j][i+1] for j = 1:n_classes]
        local_eb = repeat(local_eb, outer=[1, n_eb])
        plot_data = hcat(plot_data, local_eb)
        push!(cutoffs, cutoffs[end] + n_eb)
    end

    p = plot(
        plot_data',
        linestyle=linestyles,
        linewidth=linewidths,
        labels=reshape(class_labels, 1, length(class_labels)),
        color_palette=COLORSCHEME,
    )
    vline!(
        cutoffs,
        linewidth=linewidths,
        linestyle=:dash,
    )
    plot!(
        size=DOUBLE_WIDE,
        yformatter=y_formatter,
        fontfamily=FONTFAMILY,
        legend=:outerright,
        dpi=DPI,
    )

    # xlabel!("Training Class")
    ylabel!("Testing Accuracy")
    # xticks!(collect(1:length(local_labels)), local_labels)

    return p, plot_data
end

"""
    create_complex_condensed_plot_alt(y_hat, vals, class_labels)

Create and return an alternate complex condensed scenario plot.
"""
function create_complex_condensed_plot_alt(perfs, vals, class_labels, percentages=true)
    # Reshape the labels string vector for plotting
    local_labels = reshape(class_labels, 1, length(class_labels))
    # Determine if plotting percentages or [0, 1]
    y_formatter = percentages ? percentage_formatter : :auto
    # Set all the linewidths
    linewidths = CONDENSED_LINEWIDTH
    # Infer the number of classes
    n_classes = length(class_labels)
    # Number of experience block sample points
    n_eb = N_EB
    # Initialize cutoff x-locations (EB-LB boundaries)
    cutoffs = []
    # First EB
    push!(cutoffs, n_eb)

    # Old plot data
    for i = 1:n_classes
        # Append the training length to the cutoff
        push!(cutoffs, cutoffs[end] + size(vals[i], 2))
        # Append the number of experience block "samples"
        push!(cutoffs, cutoffs[end] + n_eb)
    end

    # Just current training data
    training_vals = []
    x_training_vals = []
    tick_locations = []
    start_point = cutoffs[1]
    for i = 1:n_classes
        # Fencepost evaluation values
        local_vals = vals[i][i, :]
        push!(local_vals, vals[i][i, end])
        n_local_vals = length(local_vals)
        # Add the tick locations as midway along training
        push!(tick_locations, start_point + floor(Int, n_local_vals/2))
        # Add the local training vals
        push!(training_vals, local_vals)
        # Add the start and stop points of the training vals
        push!(x_training_vals, collect(start_point:start_point + n_local_vals - 1))
        # Reset the start point
        start_point += n_local_vals + n_eb - 1
    end

    # Get evaluation lines locations
    fcut = vcat(0, cutoffs)
    eval_points = [mean([fcut[i-1], fcut[i]]) for i = 2:2:length(fcut)]

    # Local colors
    local_palette = palette(COLORSCHEME)

    # New training plotlines
    p = plot(
        x_training_vals,
        training_vals,
        linestyle=:solid,
        linewidth=linewidths,
        labels=local_labels,
        color_palette=local_palette,
    )

    # Vertical lines
    vline!(
        p,
        fcut,
        linewidth=linewidths,
        linestyle=:solid,
        fillalpha=0.1,
        color=:gray25,
        label="",
    )

    # The biggest headache in the world
    local_colors = [local_palette[1]; local_palette[collect(2:n_classes+1)]]
    # Eval lines
    plot!(
        p,
        eval_points,
        markershape=:circle,
        markersize=3,
        hcat(perfs...),
        color_palette=local_colors,
        linewidth=linewidths,
        linestyle=:dot,
        # linestyle=:dash,
        labels=""
    )

    # Vertical spans (gray boxes)
    vspan!(
        p,
        fcut,           # Full cutoff locations, including 0
        color=:gray25,  # 25% gray from Colors.jl
        fillalpha=0.1,  # Opacity
        label="",       # Keeps the spans out of the legend
    )

    # Format the plot
    plot!(
        size=DOUBLE_WIDE,
        yformatter=y_formatter,
        fontfamily=FONTFAMILY,
        legend=:outerright,
        legendfontsize=25,
        thickness_scaling=1,
        dpi=DPI,
        xticks=(tick_locations, class_labels),
        left_margin = 10Plots.mm,
    )

    # xlabel!("Training Class")
    ylabel!("Testing Accuracy")
    # xticks!(collect(1:length(local_labels)), local_labels)

    return p, training_vals, x_training_vals
end

# -----------------------------------------------------------------------------
# EXPERIMENTS
# -----------------------------------------------------------------------------

"""
    shuffled_mc(d::Dict, data::DataSplit, opts::opts_DDVFA)

Runs a single Monte Carlo simulation of training/testing on shuffled samples.
"""
function shuffled_mc(d::Dict, data::DataSplit, opts::opts_DDVFA)
    # Infer the number of classes
    n_classes = length(unique(data.train_y))

    # Get the random seed for the experiment
    seed = d["seed"]

    # Create the DDVFA module and setup the config
    ddvfa = DDVFA(opts)
    ddvfa.opts.display = false
    ddvfa.config = DataConfig(0, 1, 128)

    # Shuffle the data with a new random seed
    Random.seed!(seed)
    i_train = randperm(length(data.train_y))
    data.train_x = data.train_x[:, i_train]
    data.train_y = data.train_y[i_train]

    # Train and test in batch
    y_hat_train = train!(ddvfa, data.train_x, y=data.train_y)
    y_hat = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf = performance(y_hat_train, data.train_y)
    test_perf = performance(y_hat, data.test_y)

    # Save the number of F2 nodes and total categories per class
    n_F2, n_categories = get_n_categories(ddvfa)
    n_F2_sum = sum(n_F2)
    n_categories_sum = sum(n_categories)

    # Get the normalized confusion Matrix
    norm_cm = get_normalized_confusion(data.test_y, y_hat, n_classes)

    # Get the train/test accuracies
    train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)

    # Deepcopy the simulation dict and add results entries
    fulld = deepcopy(d)
    fulld["p_tr"] = train_perf
    fulld["p_te"] = test_perf
    fulld["n_F2"] = n_F2
    fulld["n_w"] = n_categories
    fulld["n_F2_sum"] = n_F2_sum
    fulld["n_w_sum"] = n_categories_sum
    fulld["norm_cm"] = norm_cm
    fulld["a_tr"] = train_accuracies
    fulld["a_te"] = test_accuracies

    # Save the results dictionary
    sim_save_name = sweep_results_dir(savename(d, "jld2"))
    @info "Worker $(myid()): saving to $(sim_save_name)"
    # wsave(sim_save_name, f)
    tagsave(sim_save_name, fulld)
end

"""
    permuted(d::Dict, data::DataSplit, opts::opts_DDVFA)

Runs a single Monte Carlo simulation of training/testing on shuffled samples.
"""
function permuted(d::Dict, data_indexed::DataSplitIndexed, opts::opts_DDVFA)
    # Get the train/test order for the experiment
    order = d["order"]

    # Create the DDVFA module and setup the config
    ddvfa = DDVFA(opts)
    ddvfa.opts.display = false
    ddvfa.config = DataConfig(0, 1, 128)

    # Get a deindexed dataset with the indexed order
    local_data = get_deindexed_data(data_indexed, order)

    # Train and test in batch
    y_hat_train = train!(ddvfa, local_data.train_x, y=local_data.train_y)
    y_hat = AdaptiveResonance.classify(ddvfa, local_data.test_x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf = performance(y_hat_train, local_data.train_y)
    test_perf = performance(y_hat, local_data.test_y)

    # Save the number of F2 nodes and total categories per class
    n_F2, n_categories = get_n_categories(ddvfa)
    n_F2_sum = sum(n_F2)
    n_categories_sum = sum(n_categories)

    # Get the normalized confusion Matrix
    norm_cm = get_normalized_confusion(local_data.test_y, y_hat, n_classes)

    # Get the train/test accuracies
    train_accuracies, test_accuracies = get_tt_accuracies(local_data, y_hat_train, y_hat, n_classes)

    # Deepcopy the simulation dict and add results entries
    fulld = deepcopy(d)
    fulld["p_tr"] = train_perf
    fulld["p_te"] = test_perf
    fulld["n_F2"] = n_F2
    fulld["n_w"] = n_categories
    fulld["n_F2_sum"] = n_F2_sum
    fulld["n_w_sum"] = n_categories_sum
    fulld["norm_cm"] = norm_cm
    fulld["a_tr"] = train_accuracies
    fulld["a_te"] = test_accuracies

    # Save the results dictionary
    saved = deepcopy(d)
    saved["perm"] = join(order)
    # sim_save_name = sweep_results_dir(savename(d, "jld2"))
    sim_save_name = sweep_results_dir(savename(saved, "jld2"))
    @info "Worker $(myid()): saving to $(sim_save_name)"
    # wsave(sim_save_name, f)
    tagsave(sim_save_name, fulld)

    # return fulld
end

"""
    unsupervised_mc(d::Dict, data::DataSplitCombined, opts::opts_DDVFA)

Runs a single Monte Carlo simulation of supervised training and unsupervised training/testing.
"""
function unsupervised_mc(d::Dict, data::DataSplitCombined, opts::opts_DDVFA)
    # Infer the number of classes
    n_classes = length(unique(data.train_y))

    # Get the random seed for the experiment
    seed = d["seed"]

    # Create the DDVFA module and setup the config
    ddvfa = DDVFA(opts)
    ddvfa.opts.display = false
    ddvfa.config = DataConfig(0, 1, 128)

    # Shuffle the data with a new random seed
    Random.seed!(seed)
    i_train = randperm(length(data.train_y))
    data.train_x = data.train_x[:, i_train]
    data.train_y = data.train_y[i_train]

    n_samples = length(data.train_y)
    i_split = Int(floor(0.8*n_samples))

    local_train_x = data.train_x[:, 1:i_split]
    local_train_y = data.train_y[1:i_split]

    local_val_x = data.train_x[:, i_split+1:end]
    local_val_y = data.train_y[i_split+1:end]

    # --- SUPERVISED ---

    # Train and test in batch
    y_hat_train = train!(ddvfa, local_train_x, y=local_train_y)
    y_hat = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf = performance(y_hat_train, local_train_y)
    test_perf = performance(y_hat, data.test_y)

    # --- UNSUPERVISED ---

    # Train in batch, unsupervised
    # y_hat_train_val = train!(ddvfa, data.val_x, y=data.val_y)
    y_hat_train_val = train!(ddvfa, local_val_x)
    # If the category is not in 1:6, replace the label as 7 for the new/incorrect bin
    replace!(x -> !(x in collect(1:n_classes)) ? 7 : x, ddvfa.labels)
    replace!(x -> !(x in collect(1:n_classes)) ? 7 : x, y_hat_train_val)
    y_hat_val = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf_val = performance(y_hat_train_val, local_val_y)
    test_perf_val = performance(y_hat, data.test_y)

    # Save the number of F2 nodes and total categories per class
    n_F2, n_categories = get_n_categories(ddvfa)
    n_F2_sum = sum(n_F2)
    n_categories_sum = sum(n_categories)

    # Get the normalized confusion Matrix
    norm_cm = get_normalized_confusion(data.test_y, y_hat, n_classes)

    # Get the train/test accuracies
    # train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    # train_accuracies_val, test_accuracies_val = get_tt_accuracies(data, y_hat_train_val, y_hat_val, n_classes)
    # TRAIN: Get the percent correct for each class
    train_accuracies = get_accuracies(local_train_y, y_hat_train, n_classes)
    # TEST: Get the percent correct for each class
    test_accuracies = get_accuracies(data.test_y, y_hat, n_classes)
    # TRAIN: Get the percent correct for each class
    train_accuracies_val = get_accuracies(local_val_y, y_hat_train_val, n_classes+1)
    # TEST: Get the percent correct for each class
    test_accuracies_val = get_accuracies(data.test_y, y_hat_val, n_classes+1)

    # Deepcopy the simulation dict and add results entries
    fulld = deepcopy(d)
    fulld["p_tr"] = train_perf
    fulld["p_te"] = test_perf
    fulld["p_trv"] = train_perf_val
    fulld["p_tev"] = test_perf_val
    fulld["n_F2"] = n_F2
    fulld["n_w"] = n_categories
    fulld["n_F2_sum"] = n_F2_sum
    fulld["n_w_sum"] = n_categories_sum
    fulld["norm_cm"] = norm_cm
    fulld["a_tr"] = train_accuracies
    fulld["a_te"] = test_accuracies
    fulld["a_trv"] = train_accuracies_val
    fulld["a_tev"] = test_accuracies_val

    # Save the results dictionary
    sim_save_name = sweep_results_dir(savename(d, "jld2"))
    @info "Worker $(myid()): saving to $(sim_save_name)"
    # wsave(sim_save_name, f)
    tagsave(sim_save_name, fulld)
end


# """
#     permuted(d::Dict, data::DataSplit, opts::opts_DDVFA)

# Runs a single Monte Carlo simulation of training/testing on shuffled samples.
# """
# function permuted(d::Dict, data_indexed::DataSplitIndexed, opts::opts_DDVFA)
#     # Get the train/test order for the experiment
#     order = d["order"]

#     # Create the DDVFA module and setup the config
#     ddvfa = DDVFA(opts)
#     ddvfa.opts.display = false
#     ddvfa.config = DataConfig(0, 1, 128)

#     # Get a deindexed dataset with the indexed order
#     data = get_deindexed_data(data_indexed, order)

#     # Train and test in batch
#     y_hat_train = train!(ddvfa, data.train_x, y=data.train_y)
#     y_hat = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

#     # Calculate performance on training data, testing data, and with get_bmu
#     train_perf = performance(y_hat_train, data.train_y)
#     test_perf = performance(y_hat, data.test_y)

#     # Save the number of F2 nodes and total categories per class
#     n_F2, n_categories = get_n_categories(ddvfa)
#     n_F2_sum = sum(n_F2)
#     n_categories_sum = sum(n_categories)

#     # Get the normalized confusion Matrix
#     norm_cm = get_normalized_confusion(data.test_y, y_hat, n_classes)

#     # Get the train/test accuracies
#     train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)

#     # Deepcopy the simulation dict and add results entries
#     fulld = deepcopy(d)
#     fulld["p_tr"] = train_perf
#     fulld["p_te"] = test_perf
#     fulld["n_F2"] = n_F2
#     fulld["n_w"] = n_categories
#     fulld["n_F2_sum"] = n_F2_sum
#     fulld["n_w_sum"] = n_categories_sum
#     fulld["norm_cm"] = norm_cm
#     fulld["a_tr"] = train_accuracies
#     fulld["a_te"] = test_accuracies

#     # Save the results dictionary
#     saved = deepcopy(d)
#     saved["perm"] = join(order)
#     # sim_save_name = sweep_results_dir(savename(d, "jld2"))
#     sim_save_name = sweep_results_dir(savename(saved, "jld2"))
#     @info "Worker $(myid()): saving to $(sim_save_name)"
#     # wsave(sim_save_name, f)
#     tagsave(sim_save_name, fulld)

#     # return fulld
# end
