"""
    lib_c3.jl

# Description
DocStringExtensions templates and common docstring constants.
"""

# -----------------------------------------------------------------------------
# DOCSTRING TEMPLATES
# -----------------------------------------------------------------------------

# Constants template
@template CONSTANTS =
"""
$(FUNCTIONNAME)

# Description
$(DOCSTRING)
"""

# Types template
@template TYPES =
"""
$(TYPEDEF)

# Summary
$(DOCSTRING)

# Fields
$(TYPEDFIELDS)
"""

# Template for functions, macros, and methods (i.e., constructors)
@template (FUNCTIONS, METHODS, MACROS) =
"""
$(TYPEDSIGNATURES)

# Summary
$(DOCSTRING)

# Method List / Definition Locations
$(METHODLIST)
"""

# -----------------------------------------------------------------------------
# COMMON DOCSTRING VARIABLES
# -----------------------------------------------------------------------------

"""
Docstring prefix denoting that the constant is used as a common docstring element for other docstrings.
"""
const COMMON_DOC = "Common docstring:"

"""
$COMMON_DOC argument for the class labels as strings used for plot axes.
"""
const ARG_CLASS_LABELS = """
- `class_labels::Vector{String}`: the string labels to use for the plot axes.
"""

"""
$COMMON_DOC argument for the [`DataSplit`](@ref) used for training, plotting, etc.
"""
const ARG_DATA_SPLIT = """
- `data::DataSplit`: the original dataset with a train, val, and test split.
"""

"""
$COMMON_DOC argument for a set of features as a 2-D matrix.
"""
const ARG_DATA_MATRIX = """
- `data::RealMatrix`: the data as a 2-D matrix of real values.
"""

"""
$COMMON_DOC argument for the true target values.
"""
const ARG_Y = """
- `y::IntegerVector`: the true targets as integers.
"""

"""
$COMMON_DOC argument for the classifier's target outputs.
"""
const ARG_Y_HAT = """
- `y_hat::IntegerVector`: the approximated targets generated by the classifier.
"""

"""
$COMMON_DOC argument for the target estimates on the training data.
"""
const ARG_Y_HAT_TRAIN = """
- `y_hat_train::IntegerVector`: the classifier estimates from the training data.
"""

"""
$COMMON_DOC argument for classifier validation data estimates.
"""
const ARG_Y_HAT_VAL = """
- `y_hat_val::IntegerVector`: the classifier estimates from the validation data.
"""

"""
$COMMON_DOC argument flag to use a custom percentage formatter during plotting.
"""
const ARG_PERCENTAGES = """
- `percentages::Bool=false`: optional, flag to use the custom percentage formatter or not.
"""

"""
$COMMON_DOC the y-lim bounds for a plot.
"""
const ARG_BOUNDS = """
- `bounds::Tuple{Float, Float}=$PERCENTAGES_BOUNDS`: optional, the bounds for the y-lim bounds of the plot.
"""

"""
$COMMON_DOC argument for the directory string to save to.
"""
const DOC_ARG_SAVE_DIR = """
- `dir::AbstractString`: the directory to save to.
"""