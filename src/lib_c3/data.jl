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
A single dataset of [`Features`](@ref), [`Targets`](@ref), and human-readable string [`Labels`](@ref).
"""
struct LabeledDataset
    """
    Collection of [`Features`](@ref) in the labeled dataset.
    """
    x::Features

    """
    [`Targets`](@ref) corresponding to the [`Features`](@ref).
    """
    y::Targets

    """
    Human-readable [`Labels`](@ref) corresponding to the [`Targets`](@ref) values.
    """
    labels::Labels
end

"""
A single dataset of vectored labeled data with [`Features`](@ref), [`Targets`](@ref), and human-readable string [`Labels`](@ref).
"""
struct VectorLabeledDataset
    """
    A vector of [`Features`](@ref) matrices.
    """
    x::Vector{Features}

    """
    A vector of [`Targets`](@ref) corresponding to the [`Features`](@ref).
    """
    y::Vector{Targets}

    """
    String [`Labels`](@ref) corresponding to the [`Targets`](@ref).
    """
    labels::Labels
end

"""
A basic struct for encapsulating the components of supervised training.
"""
struct DataSplit <: MatrixData
    """
    Training [`LabeledDataset`](@ref).
    """
    train::LabeledDataset

    """
    Validation [`LabeledDataset`](@ref).
    """
    val::LabeledDataset

    """
    Test [`LabeledDataset`](@ref).
    """
    test::LabeledDataset
end

"""
A struct for encapsulating the components of supervised training in vectorized form.
"""
struct DataSplitIndexed <: VectoredData
    """
    Training [`VectorLabeledDataset`](@ref).
    """
    train::VectorLabeledDataset

    """
    Validation [`VectorLabeledDataset`](@ref).
    """
    val::VectorLabeledDataset

    """
    Test [`VectorLabeledDataset`](@ref).
    """
    test::VectorLabeledDataset
end

"""
A struct for combining training and validation data, containing only train and test splits.
"""
struct DataSplitCombined <: MatrixData
    """
    Training [`LabeledDataset`](@ref).
    """
    train::LabeledDataset

    """
    Testing [`LabeledDataset`](@ref).
    """
    test::LabeledDataset
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
A constructor for a [`LabeledDataset`](@ref) that merges two other [`LabeledDataset`](@ref)s.

# Arguments
- `d1::LabeledDataset`: the first [`LabeledDataset`](@ref) to consolidate.
- `d2::LabeledDataset`: the second [`LabeledDataset`](@ref) to consolidate.
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
- `data::DataSplit`: the [`DataSplit`](@ref) struct for consolidating validation [`Features`](@ref) and [`Labels`](@ref) into the training data.
"""
function DataSplitCombined(data::DataSplit)
    # Consolidate trainind and validation, and return the struct in one step
    return DataSplitCombined(
        LabeledDataset(data.train, data.val),
        data.test,
    )
end
