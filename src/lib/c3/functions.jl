"""
    functions.jl

# Description
This file contains the majority of experiment functions for the DCCR project.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Returns the sigmoid function on x.

# Arguments
- `x::Real`: the float or int to compute the sigmoid function upon.
"""
function sigmoid(x::Real)
    return one(x) / (one(x) + exp(-x))
end

"""
Returns the activations from a single directory.

# Arguments
- `data_dir::AbstractString`: the single data directory to load the features from.
"""
function collect_activations(data_dir::AbstractString)
    data_full = readdlm(joinpath(data_dir, "average_features.csv"), ',')
    return data_full
end

"""
Return just the yolo activations from a list of data directories.

# Arguments
- `data_dirs::AbstractArray`: the data directories to load the yolo activations from.
- `cell::Integer`: the number of cells corresponding to the windowing procedure.
"""
function collect_all_activations(data_dirs::AbstractArray, cell::Integer)
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
Return the yolo activations, training targets, and condensed labels list from a list of data directories.

# Arguments
- `data_dirs::Vector{String}`: the directories to load the data from.
- `cell::Integer`: the number of cells to use in the windowed averaging procedure.
"""
function collect_all_activations_labeled(data_dirs::Vector{String}, cell::Integer)
    # The final dimension is 128 (YOLOv3 feature size) times the number of cells
    top_dim = 128*cell
    # Initialized the output dataset, targets, and their string labels
    data_grand = Matrix{Float64}(undef, top_dim, 0)
    targets = Vector{Int64}()
    labels = Vector{String}()
    # Iterate over each data directory
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

    # Return the final dataset, its target values, and the string labels
    return data_grand, targets, labels
end

"""
Get the distribution parameters for preprocessing.

# Arguments
- `data::RealMatrix`: a 2-D matrix of features for computing the Gaussian statistics of.
"""
function get_dist(data::RealMatrix)
    return fit(ZScoreTransform, data, dims=2)
end

"""
Preprocesses one dataset of features, scaling and squashing along the feature axes.

# Arguments
- `dt::ZScoreTransform`: the Gaussian statistics of the features.
- `scaling::Real`: the sigmoid scaling parameter.
- `data::RealMatrix`: the 2-D matrix of features to transform.
"""
function feature_preprocess(dt::ZScoreTransform, scaling::Real, data::RealMatrix)
    # Normalize the dataset via the ZScoreTransform
    new_data = StatsBase.transform(dt, data)
    # Squash the data sigmoidally with the scaling parameter
    new_data = sigmoid.(scaling*new_data)
    # Return the normalized and scaled data
    return new_data
end

"""
Load the orbits data and preprocess the features.

# Arguments
- `data_dir::AbstractString`: the top-level data directory.
- `data_dirs::Vector{String}`: the subfolders to load.
- `scaling::Real`: the sigmoidal scaling parameter to use.
"""
function load_orbits(
    data_dir::AbstractString,
    data_dirs::Vector{String},
    scaling::Real
)
    # Point to the train, validation, and test block folders
    train_dir = joinpath(data_dir, "LBs")
    val_dir = joinpath(data_dir, "Val")
    test_dir = joinpath(data_dir, "EBs")

    # Point to all of the subfolders in each data block
    train_data_dirs = [joinpath(train_dir, data_dir) for data_dir in data_dirs]
    val_data_dirs = [joinpath(val_dir, data_dir) for data_dir in data_dirs]
    test_data_dirs = [joinpath(test_dir, data_dir) for data_dir in data_dirs]

    # Collect all of the labeled activations
    train_x, train_y, train_labels = collect_all_activations_labeled(train_data_dirs, 1)
    val_x, val_y, val_labels = collect_all_activations_labeled(val_data_dirs, 1)
    test_x, test_y, test_labels = collect_all_activations_labeled(test_data_dirs, 1)

    # Get the distribution parameters for the training data
    dt = get_dist(train_x)

    # Preprocess all of the data based upon the statistics of the training data
    train_x = feature_preprocess(dt, scaling, train_x)
    val_x = feature_preprocess(dt, scaling, val_x)
    test_x = feature_preprocess(dt, scaling, test_x)

    # Construct the split dataset struct
    data_struct = DataSplit(
        # Training
        LabeledDataset(
            train_x,
            train_y,
            train_labels,
        ),
        # Validation
        LabeledDataset(
            val_x,
            val_y,
            val_labels,
        ),
        # Testing
        LabeledDataset(
            test_x,
            test_y,
            test_labels
        ),
    )

    # Return the single object containing the data
    return data_struct
end

"""
Shuffles the training orbits data.

# Arguments
- `data::DataSplit`: the [`DataSplit`](@ref) orbits data coming from `load_orbits`.
"""
function shuffle_orbits(data::DataSplit)
    # Get a random set of indices and shuffle both x and y
    i_train = randperm(length(data.train.y))
    local_x = data.train.x[:, i_train]
    local_y = data.train.y[i_train]

    # Construct the split dataset struct
    data_struct = DataSplit(
        # Training
        LabeledDataset(
            local_x,
            local_y,
            data.train.labels,
        ),
        data.val,
        data.test
    )

    return data_struct
end

"""
Create a DataSplitIndexed object from a DataSplit.

# Arguments
- `data::DataSplit`: the DataSplit to separate into vectors of matrices.
"""
function get_indexed_data(data::DataSplit)
    # Assume the same number of classes in each category
    n_classes = length(unique(data.train.y))

    # Construct empty fields
    train_x = Vector{Matrix{Float}}()
    train_y = Vector{Vector{Int}}()
    train_labels = Vector{String}()
    val_x = Vector{Matrix{Float}}()
    val_y = Vector{Vector{Int}}()
    val_labels = Vector{String}()
    test_x = Vector{Matrix{Float}}()
    test_y = Vector{Vector{Int}}()
    test_labels = Vector{String}()

    # Iterate over every class
    for i = 1:n_classes
        i_train = findall(x -> x == i, data.train.y)
        push!(train_x, data.train.x[:, i_train])
        push!(train_y, data.train.y[i_train])
        i_val = findall(x -> x == i, data.val.y)
        push!(val_x, data.val.x[:, i_val])
        push!(val_y, data.val.y[i_val])
        i_test = findall(x -> x == i, data.test.y)
        push!(test_x, data.test.x[:, i_test])
        push!(test_y, data.test.y[i_test])
    end

    train_labels = data.train.labels
    val_labels = data.val.labels
    test_labels = data.test.labels

    # Construct the indexed data split
    data_indexed = DataSplitIndexed(
        VectorLabeledDataset(
            train_x,
            train_y,
            train_labels,
        ),
        VectorLabeledDataset(
            val_x,
            val_y,
            val_labels,
        ),
        VectorLabeledDataset(
            test_x,
            test_y,
            test_labels,
        ),
    )
    return data_indexed
end

"""
Turn a DataSplitIndexed into a DataSplit with the given train/test order.

# Arguments
- `data::DataSplitIndexed`: the indexed data to consolidate back into a DataSplit.
- `order::IntegerVector`: the order used by the indexed data for correctly deindexing.
"""
function get_deindexed_data(data::DataSplitIndexed, order::IntegerVector)
    dim = 128
    train_x = Array{Float64}(undef, dim, 0)
    train_y = Array{Int}(undef, 0)
    train_labels = Vector{String}()

    val_x = Array{Float64}(undef, dim, 0)
    val_y = Array{Int}(undef, 0)
    val_labels = Vector{String}()

    test_x = Array{Float64}(undef, dim, 0)
    test_y = Array{Int}(undef, 0)
    test_labels = Vector{String}()

    for i in order
        train_x = hcat(train_x, data.train.x[i])
        train_y = vcat(train_y, data.train.y[i])
        val_x = hcat(val_x, data.val.x[i])
        val_y = vcat(val_y, data.val.y[i])
        test_x = hcat(test_x, data.test.x[i])
        test_y = vcat(test_y, data.test.y[i])
    end

    train_labels = data.train.labels[order]
    val_labels = data.val.labels[order]
    test_labels = data.test.labels[order]

    # Construct the DataSplit
    data_struct = DataSplit(
        LabeledDataset(
            train_x,
            train_y,
            train_labels,
        ),
        LabeledDataset(
            val_x,
            val_y,
            val_labels,
        ),
        LabeledDataset(
            test_x,
            test_y,
            test_labels
        ),
    )

    return data_struct
end

"""
Map the experiment orbit names to their data directories and plotting class labels.

# Arguments
- `selection::Vector{String}`: the selection of labels corresponding to both data directories and plotting labels.
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
Wrapper method for getting the raw confusion matrix.

# Arguments
- `y::IntegerVector`: the target values.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Integer)
    return confusmat(n_classes, y, y_hat)
end

"""
Get the normalized confusion matrix.

# Arguments
- `y::IntegerVector`: the target values.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_normalized_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Integer)
    cm = get_confusion(y, y_hat, n_classes)
    # total = sum(cm, dims=1)
    total = sum(cm, dims=2)'
    norm_cm = cm./total
    return norm_cm
end

"""
Get a list of the percentage accuracies.

# Arguments
- `y::IntegerVector`: the target values.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_accuracies(y::IntegerVector, y_hat::IntegerVector, n_classes::Integer)
    cm = get_confusion(y, y_hat, n_classes)
    correct = [cm[i,i] for i = 1:n_classes]
    # total = sum(cm, dims=1)
    total = sum(cm, dims=2)'
    accuracies = correct'./total

    return accuracies
end

"""
Get two lists of the training and testing accuracies.

# Arguments
- `data::MatrixData`: the training and testing dataset, containing a vector of training and testing labels `data.train.y` and `data.test.y`.
- `y_hat_train::IntegerVector`: the training estimates.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_tt_accuracies(
    data::MatrixData,
    y_hat_train::IntegerVector,
    y_hat::IntegerVector,
    n_classes::Integer
)
    # TRAIN: Get the percent correct for each class
    train_accuracies = get_accuracies(data.train.y, y_hat_train, n_classes)

    # TEST: Get the percent correct for each class
    test_accuracies = get_accuracies(data.test.y, y_hat, n_classes)

    # Return the list of accuracy values for each class in training and testing
    return train_accuracies, test_accuracies
end

"""
Returns both the number of F2 categories and total number of weights per class as two lists.

# Arguments
- `ddvfa::DDVFA`: the DDVFA module to calculate the statistics for.
- `n_classes::Int`: the number of target classes that the model was trained upon.
"""
function get_n_categories(ddvfa::DDVFA, n_classes::Int)
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
Wrapper of `stratifiedobs`, returns a manual train/test x/y split from a data matrix and labels using MLDataUtils.

# Arguments
- `data::RealMatrix`: the feature data to split into training and testing.
- `targets::IntegerVector`: the labels corresponding to the data to split into training and testing.
"""
function get_manual_split(data::RealMatrix, targets::IntegerVector)
    (X_train, y_train), (X_test, y_test) = stratifiedobs((data, targets))
    return (X_train, y_train), (X_test, y_test)
end

"""
Convert a column of lists in a DataFrame into a matrix for analysis.

# Arguments
- `df::DataFrame`: the DataFrame containing the column of lists.
- `row::Symbol`: the symbolic name of the row in the DataFrame to convert into a matrix.
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
Common docstring, argument for the class labels as strings used for plot axes.
"""
const ARG_CLASS_LABELS = """
- `class_labels::Vector{String}`: the string labels to use for the plot axes.
"""

"""
Common docstring, argument for the [`DataSplit`](@ref) used for training, plotting, etc.
"""
const ARG_DATA_SPLIT = """
- `data::DataSplit`: the original dataset with a train, val, and test split.
"""

"""
Common docstring, argument for a set of features as a 2-D matrix.
"""
const ARG_DATA_MATRIX = """
- `data::RealMatrix`: the data as a 2-D matrix of real values.
"""

"""
Common docstring, argument for the true target values.
"""
const ARG_Y = """
- `y::IntegerVector`: the true targets as integers.
"""

"""
Common docstring, argument for the classifier's target outputs.
"""
const ARG_Y_HAT = """
- `y_hat::IntegerVector`: the approximated targets generated by the classifier.
"""

"""
Common docstring, argument for the target estimates on the training data.
"""
const ARG_Y_HAT_TRAIN = """
- `y_hat_train::IntegerVector`: the classifier estimates from the training data.
"""

const ARG_Y_HAT_VAL = """
- `y_hat_val::IntegerVector`: the classifier estimates from the validation data.
"""

const ARG_PERCENTAGES = """
- `percentages::Bool=false`: optional, flag to use the custom percentage formatter or not.
"""

"""
Returns a handle to a labeled and annotated heatmap plot of the confusion matrix.

# Arguments
$ARG_CLASS_LABELS
$ARG_Y
$ARG_Y_HAT
"""
function create_confusion_heatmap(
    class_labels::Vector{String},
    y::IntegerVector,
    y_hat::IntegerVector
)
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
            i-0.5,
            j-0.5,
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
Returns a handle to a labeled and annotated heatmap plot of the confusion matrix.

# Arguments
$ARG_CLASS_LABELS
- `norm_cm::RealMatrix`: the normalized confuction matrix to plot as a heatmap.
"""
function create_custom_confusion_heatmap(
    class_labels::Vector{String},
    norm_cm::RealMatrix
)
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
Return a grouped bar chart with class accuracies.

# Arguments
$ARG_DATA_SPLIT
$ARG_Y_HAT_TRAIN
$ARG_Y_HAT
$ARG_CLASS_LABELS
$ARG_PERCENTAGES
"""
function create_accuracy_groupedbar(
    data::DataSplit,
    y_hat_train::IntegerVector,
    y_hat::IntegerVector,
    class_labels::Vector{String} ;
    percentages::Bool=false
)
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
Return a grouped bar chart with comparison bars.

# Arguments
$ARG_DATA_SPLIT
$ARG_Y_HAT_VAL
$ARG_Y_HAT
$ARG_CLASS_LABELS
$ARG_PERCENTAGES
- `extended::Bool=false`: if the plot needs to be extended to another category, compensating for misclassification.
- ``
"""
function create_comparison_groupedbar(
    data::DataSplit,
    y_hat_val::IntegerVector,
    y_hat::IntegerVector,
    class_labels::Vector{String} ;
    percentages=false,
    extended=false
)
    # Infer the number of classes from the class labels
    n_classes = length(class_labels)
    # If we need a category to dump in, extend the
    # extended && n_classes+= 1
    if extended
        n_classes += 1
    end

    # Get the training and testing accuracies
    # train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    test_accuracies = get_accuracies(data.test.y, y_hat, n_classes)
    test_accuracies_val = get_accuracies(data.test.y, y_hat_val, n_classes)

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
Return a colored and formatted boxplot of the data.

# Arguments
$ARG_DATA_MATRIX
$ARG_CLASS_LABELS
$ARG_PERCENTAGES
- `bounds::Tuple{Float, Float}=$PERCENTAGES_BOUNDS`: optional, the bounds for the y-lim bounds of the plot.
- `violin_bandwidth::Real=0.01`: the bandwidth parameter passed to the violin plot.
"""
function create_boxplot(
    data::RealMatrix,
    class_labels::Vector{String} ;
    percentages=false,
    bounds::Tuple{Float, Float}=PERCENTAGES_BOUNDS,
    violin_bandwidth::Real=0.01
)
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
        bandwidth=violin_bandwidth,
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
        # Set the bounds
        # ylims!(p, (-Inf, 1))
        # ylims!(p, (0.6, 1))
        ylims!(p, bounds)
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
Return a colored and formatted boxplot of the data.
"""
function create_inverted_boxplot(data::RealMatrix, class_labels::Vector{String} ; percentages=false, bounds_override=[])
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

    # if percentages
    #     # ylims!(p, (-Inf, 1))
    #     # ylims!(p, (0.6, 1))
    #     ylims!(p, PERCENTAGES_BOUNDS)
    # end

    if percentages
        # Set the bounds
        if !isempty(bounds_override)
            local_bounds = bounds_override
        else
            local_bounds = PERCENTAGES_BOUNDS
        end
        # ylims!(p, (-Inf, 1))
        # ylims!(p, (0.6, 1))
        ylims!(p, local_bounds)
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
Create and return a simplified condensed scenario plot.
"""
function create_condensed_plot(perfs, class_labels, percentages::Bool=true)
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
Create and return a complex condensed scenario plot.
"""
function create_complex_condensed_plot(perfs, vals, class_labels, percentages::Bool=true)
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
Create and return an alternate complex condensed scenario plot.
"""
function create_complex_condensed_plot_alt(perfs, vals, class_labels, percentages::Bool=true)
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
Runs a single Monte Carlo simulation of training/testing on shuffled samples.

# Arguments
- `d::Dict`: a logging dictionary storing simulation parameters.
- `data::DataSplit`: a train/test split of features and labels.
- `opts::opts_DDVFA`: the options for DDVFA construction.
"""
function shuffled_mc(d::Dict, data::DataSplit, opts::opts_DDVFA)
    # Infer the number of classes
    n_classes = length(unique(data.train.y))

    # Get the random seed for the experiment
    seed = d["seed"]

    # Create the DDVFA module and setup the config
    ddvfa = DDVFA(opts)
    ddvfa.opts.display = false
    ddvfa.config = DataConfig(0, 1, 128)

    # Shuffle the data with a new random seed
    Random.seed!(seed)
    i_train = randperm(length(data.train.y))
    data.train.x = data.train.x[:, i_train]
    data.train.y = data.train.y[i_train]

    # Train and test in batch
    y_hat_train = train!(ddvfa, data.train.x, y=data.train.y)
    y_hat = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf = performance(y_hat_train, data.train.y)
    test_perf = performance(y_hat, data.test.y)

    # Save the number of F2 nodes and total categories per class
    # n_F2, n_categories = get_n_categories(ddvfa)
    n_F2, n_categories = get_n_categories(ddvfa, n_classes)
    n_F2_sum = sum(n_F2)
    n_categories_sum = sum(n_categories)

    # Get the normalized confusion Matrix
    norm_cm = get_normalized_confusion(data.test.y, y_hat, n_classes)

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
Runs a single Monte Carlo simulation of training/testing on shuffled samples.

# Arguments
- `d::Dict`: a logging dictionary storing simulation parameters.
- `data::DataSplitIndexed`: an indexed train/test split of features and labels.
- `opts::opts_DDVFA`: the options for DDVFA construction.
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
    y_hat_train = train!(ddvfa, local_data.train.x, y=local_data.train.y)
    y_hat = AdaptiveResonance.classify(ddvfa, local_data.test.x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf = performance(y_hat_train, local_data.train.y)
    test_perf = performance(y_hat, local_data.test.y)

    # Save the number of F2 nodes and total categories per class
    # n_F2, n_categories = get_n_categories(ddvfa)
    n_F2, n_categories = get_n_categories(ddvfa, 6)
    n_F2_sum = sum(n_F2)
    n_categories_sum = sum(n_categories)

    # Get the normalized confusion Matrix
    norm_cm = get_normalized_confusion(local_data.test.y, y_hat, n_classes)

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
Runs a single Monte Carlo simulation of supervised training and unsupervised training/testing.

# Arguments
- `d::Dict`: a logging dictionary storing simulation parameters.
- `data::DataSplitCombined`: a train/test split of features and labels.
- `opts::opts_DDVFA`: the options for DDVFA construction.
"""
function unsupervised_mc(d::Dict, data::DataSplitCombined, opts::opts_DDVFA)
    # Infer the number of classes
    n_classes = length(unique(data.train.y))

    # Get the random seed for the experiment
    seed = d["seed"]

    # Create the DDVFA module and setup the config
    ddvfa = DDVFA(opts)
    ddvfa.opts.display = false
    ddvfa.config = DataConfig(0, 1, 128)

    # Shuffle the data with a new random seed
    Random.seed!(seed)
    i_train = randperm(length(data.train.y))
    pre_train_x = data.train.x[:, i_train]
    pre_train_y = data.train.y[i_train]

    # Get the number of samples in the training dataset to split it up
    n_samples = length(pre_train_y)
    i_split = Int(floor(0.8*n_samples))

    # Split the original training dataset into train (supervised) and val (unsupervised)
    local_train_x = pre_train_x[:, 1:i_split]
    local_train_y = pre_train_y[1:i_split]
    local_val_x = pre_train_x[:, i_split+1:end]
    local_val_y = pre_train_y[i_split+1:end]

    # --- SUPERVISED ---

    # Train and test in batch
    y_hat_train = train!(ddvfa, local_train_x, y=local_train_y)
    y_hat = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

    # Calculate performance on training data, testing data, and with get_bmu
    train_perf = performance(y_hat_train, local_train_y)
    test_perf = performance(y_hat, data.test.y)

    # --- UNSUPERVISED ---

    # Train in batch, unsupervised
    # y_hat_train_val = train!(ddvfa, data.val.x, y=data.val.y)
    y_hat_train_val = train!(ddvfa, local_val_x)
    # If the category is not in 1:6, replace the label as 7 for the new/incorrect bin
    replace!(x -> !(x in collect(1:n_classes)) ? 7 : x, ddvfa.labels)
    replace!(x -> !(x in collect(1:n_classes)) ? 7 : x, y_hat_train_val)
    y_hat_val = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

    # Calculate performance on second training data, testing data, and with get_bmu
    train_perf_val = performance(y_hat_train_val, local_val_y)
    test_perf_val = performance(y_hat_val, data.test.y)

    # Save the number of F2 nodes and total categories per class
    # n_F2, n_categories = get_n_categories(ddvfa)
    n_F2, n_categories = get_n_categories(ddvfa, n_classes)
    n_F2_sum = sum(n_F2)
    n_categories_sum = sum(n_categories)

    # Get the normalized confusion matrices
    norm_cm = get_normalized_confusion(data.test.y, y_hat, n_classes)
    norm_cm_val = get_normalized_confusion(data.test.y, y_hat_val, n_classes + 1)

    # Get the train/test accuracies
    # train_accuracies, test_accuracies = get_tt_accuracies(data, y_hat_train, y_hat, n_classes)
    # train_accuracies_val, test_accuracies_val = get_tt_accuracies(data, y_hat_train_val, y_hat_val, n_classes)
    # TRAIN: Get the percent correct for each class
    train_accuracies = get_accuracies(local_train_y, y_hat_train, n_classes)
    # TEST: Get the percent correct for each class
    test_accuracies = get_accuracies(data.test.y, y_hat, n_classes)
    # TRAIN: Get the percent correct for each class
    train_accuracies_val = get_accuracies(local_val_y, y_hat_train_val, n_classes+1)
    # TEST: Get the percent correct for each class
    test_accuracies_val = get_accuracies(data.test.y, y_hat_val, n_classes+1)

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
    fulld["norm_cm_val"] = norm_cm_val
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

"""
The packed data directory as a DrWatson-style path function.

# Arguments
- `args...`: string arguments a subsequent file or folders.
"""
packed_dir(args...) = datadir("packed", args...)

"""
The unpacked data directory as a DrWatson-style path function.

# Arguments
- `args...`: string arguments a subsequent file or folders.
"""
unpacked_dir(args...) = datadir("unpacked", args...)

"""
Packs the data under the provided experiment name folder into an LFS-tracked tarball.

# Arguments
- `experiment_name::AbstractString`: the name of the file destination to pack from the `unpacked_dir` to the `packed_dir`.
"""
function pack_data(experiment_name::AbstractString)
    from_dir = unpacked_dir(experiment_name)
    to_file = packed_dir(experiment_name * ".tar")
    Tar.create(from_dir, to_file)
end

"""
Unpacks data at the provided experiment name tarball into a working directory.

# Arguments
- `experiment_name::AbstractString`: the name of the file to unpack from the `packed_dir` to the `unpacked_dir`.
"""
function unpack_data(experiment_name::AbstractString)
    from_file = packed_dir(experiment_name * ".tar")
    to_dir = unpacked_dir(experiment_name)
    Tar.extract(from_file, to_dir)
end

"""
If the provided experiment unpacked directory does not exist, this unpacks it.

# Arguments
- `experiment_name::AbstractString`: the name of the file to unpack from the `packed_dir` to the `unpacked_dir`.
"""
function safe_unpack(experiment_name::AbstractString)
    # If the unpacked data directory does not already exist, unpack to it
    if !isdir(unpacked_dir(experiment_name))
        unpack_data(experiment_name)
    end
end

"""
Loads the default orbit data configuration.

# Arguments
- `data_dir::AbstractString`: the relative/absolute directory containing the data.
- `scaling::Float`: the sigmoid scaling parameter, default `scaling=2.0`
"""
function load_default_orbit_data(data_dir::AbstractString ; scaling::Float=2.0)
    # Select which data entries to use for the experiment
    data_selection = [
        "dot_dusk",
        "dot_morning",
        # "emahigh_dusk",
        # "emahigh_morning",
        "emalow_dusk",
        "emalow_morning",
        "pr_dusk",
        "pr_morning",
    ]

    # Load the data names and class labels from the selection
    data_dirs, class_labels = get_orbit_names(data_selection)

    # Number of classes
    n_classes = length(data_dirs)

    # Load the data
    data = load_orbits(data_dir, data_dirs, scaling)

    # Sort/reload the data as indexed components
    data_indexed = get_indexed_data(data)

    # Return the original data, indexed data, class labels, and the number of classes for convenience
    return data, data_indexed, class_labels, data_selection, n_classes
end


# Top data directory
const data_dir = unpacked_dir("activations_yolov3_cell=1")


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
#     y_hat_train = train!(ddvfa, data.train.x, y=data.train.y)
#     y_hat = AdaptiveResonance.classify(ddvfa, data.test.x, get_bmu=true)

#     # Calculate performance on training data, testing data, and with get_bmu
#     train_perf = performance(y_hat_train, data.train.y)
#     test_perf = performance(y_hat, data.test.y)

#     # Save the number of F2 nodes and total categories per class
#     n_F2, n_categories = get_n_categories(ddvfa)
#     n_F2_sum = sum(n_F2)
#     n_categories_sum = sum(n_categories)

#     # Get the normalized confusion Matrix
#     norm_cm = get_normalized_confusion(data.test.y, y_hat, n_classes)

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
