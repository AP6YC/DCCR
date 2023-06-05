"""
    lib_c3.jl

# Description
Collects all of the C3 library code.
"""

# -----------------------------------------------------------------------------
# ABSTRACT TYPES
# -----------------------------------------------------------------------------

"""
Abstract supertype for all Data structs in this library.
"""
abstract type Data end

"""
Abstract type for Data structs that represent features as matrices.
"""
abstract type MatrixData <: Data end

"""
Abstract type for Data structs that represent features as vectors of matrices.
"""
abstract type VectoredData <: Data end

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------

"""
Definition of features as a matrix of floating-point numbers of dimension (feature_dim, n_samples).
"""
const Features = Matrix{Float}

"""
Definition of targets as a vector of integers.
"""
const Targets = Vector{Int}

"""
Definition of labels as a vector of strings.
"""
const Labels = Vector{String}

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
A single dataset of features, targets, and human-readable string labels.
"""
struct LabeledDataset
    """
    Collection of features in the labeled dataset.
    """
    x::Features

    """
    Targets corresponding to the features.
    """
    y::Targets

    """
    Human-readable labels corresponding to the target values.
    """
    labels::Labels
end

"""
A single dataset of vectored labeled data with features, targets, and human-readable string labels.
"""
struct VectorLabeledDataset
    """
    A vector of features matrices.
    """
    x::Vector{Features}

    """
    A vector of targets corresponding to the features.
    """
    y::Vector{Targets}

    """
    String labels corresponding to the targets.
    """
    labels::Labels
end

"""
A basic struct for encapsulating the components of supervised training.
"""
struct DataSplit <: MatrixData
    """
    Training dataset.
    """
    train::LabeledDataset

    """
    Validation dataset.
    """
    val::LabeledDataset

    """
    Test dataset.
    """
    test::LabeledDataset
end

"""
A basic struct for encapsulating the components of supervised training.
"""
struct DataSplitIndexed <: VectoredData
    """
    Training vectorized dataset.
    """
    train::VectorLabeledDataset

    """
    Validation vectorized dataset.
    """
    val::VectorLabeledDataset

    """
    Test vectorized dataset.
    """
    test::VectorLabeledDataset
end

"""
A struct for combining training and validation data, containing only train and test splits.
"""
struct DataSplitCombined <: MatrixData
    """
    Training dataset.
    """
    train::LabeledDataset

    """
    Testing dataset.
    """
    test::LabeledDataset
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
A constructor for LabeledDataset that merges two other LabeledDatasets.

# Arguments
- `d1::LabeledDataset`: the first LabeledDataset to consolidate.
- `d2::LabeledDataset`: the second LabeledDataset to consolidate.
"""
function LabeledDataset(d1::LabeledDataset, d2::LabeledDataset)
    # Consolidate everything and construct in one step
    return LabeledDataset(
        hcat(d1.x, d2.x),
        vcat(d1.y, d2.y),
        vcat(d1.labels, d2.labels),
    )
end

"""
Constructs a DataSplitCombined from an existing DataSplit by consolidating the training and validation data.

# Arguments
- `data::DataSplit`: the DataSplit struct for consolidating validation features and labels into the training data.
"""
function DataSplitCombined(data::DataSplit)
    # Consolidate trainind and validation, and return the struct in one step
    return DataSplitCombined(
        LabeledDataset(data.train, data.val),
        data.test,
    )
end
