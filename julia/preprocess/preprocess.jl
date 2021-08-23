
"""
    preprocess.jl

Description:
    Preprocessing toolchain for the YOLOv3 features coming into C3.

Authors:
    - Sasha Petrenko <sap625@mst.edu> <sap625@umsystem.edu>
"""

using Statistics: mean
using StatsBase

"""
    window_average(data::Array, bins::Int)

Average the 3-D data into bins.

For example, if bins = 2, then the features are averaged into two bins per dimension (i.e. four quadrants in a 2-D image), resulting in a vector of length four times the feature length.

# Arguments
- `data::Array`: the input features from C1.
- `bins::Int`: the number of cells in each dimension (see above).
"""
function window_average(data::Array, bins::Int)

    # If we are averaging multiple windows
    if bins > 1
        # Explicitly extract the shape of the data
        dim_y, dim_x, dim_feat = size(data)

        # Get the number of pixels for each dimension in each cell
        # TODO: this loses a pixel row for odd dims, make a special case for this
        split_x = Int(floor(dim_x/bins))
        split_y = Int(floor(dim_y/bins))

        # Initialize the return container (always faster to preallocate)
        average_output = zeros(Float64, dim_feat*bins^2)

        # Iterate across all of the cells/bins (indexing at 0 for later ease)
        for iy = 0:bins - 1
            for ix = 0:bins - 1
                # Get the sub array, careful of Julia's 1-indexing
                sub_array = data[
                    split_y*iy + 1 : split_y * (iy + 1),
                    split_x*ix + 1 : split_x * (ix + 1), :]
                # Compute the sub array's average features
                local_averages = mean(sub_array, dims=(1, 2))
                # Assign the averages into the output container
                feat_index = dim_feat*(iy*bins+ix) + 1
                average_output[feat_index : feat_index + dim_feat - 1] = local_averages
            end
        end
    # Otherwise, take the average along x and y all at once
    else
        average_output = vec(mean(data, dims=(1, 2)))
    end

    return average_output
end # window_average(data::Array, bins::Int)

"""
    sigmoid(x::Real)

Return the sigmoid function on x.

# Arguments
- `x::Real`: the value to process through the sigmoid function.
"""
function sigmoid(x::Real)
    return one(x) / (one(x) + exp(-x))
end # sigmoid(x::Real)

"""
    sigmoid(x::Array)

Broadcasts sigmoid(x::Real) across an array x.

# Arguments
- `x::Array`: array to process sigmoidally.
"""
function sigmoid(x::Array)
    return sigmoid.(x)
end # sigmoid(x::Array)

"""
    task_feature_preprocess(x::Array)

Preprocess the incoming features for task detection module 3.

# Arguments
- `x::Array`: array to preprocess for C3.
- `params::TaskDetectorParameters`: the algorithm's internal parameters.
"""
function task_feature_preprocess(x::Array, params::TaskDetectorParameters)
    # Average the data into windows
    # TODO: set dim in config
    x_norm = window_average(x, params.windows)

    # Standardize with the data transformer
    x_norm = StatsBase.transform(params.transformer, x_norm)

    # Sigmoidally squash the data to normalize between [0, 1] scaled by sigmoid_scaling
    x_norm = sigmoid(params.sigmoid_scaling.*(x_norm.*2 .- 1))

    return x_norm
end # task_feature_preprocess(x::Array)

# Feature preprocessing configuration
const MEAN_FILE = "../config/mean/1.csv"
const SCALE_FILE = "../config/scale/1.csv"
const WINDOWS = 1
const FEATURE_DIM = 128
const SIGMOID_SCALING = 3.0

# Load the mean and scale local configs
mean = vec(readdlm(joinpath(local_path, mean_file), ',', Float64, '\n'))
scale = vec(readdlm(joinpath(local_path, scale_file), ',', Float64, '\n'))

# Construct the transformer from the mean and scale parameters
transformer = ZScoreTransform(windows*feat_dim, 2, mean, scale)

"""
    PreprocessParameters

Struct containing the options for the taskdetector.

# Fields
- `mean::Array{Float64, 1}`: means of each feature, used by transformer.
- `scale::Array{Float64, 1}`: scaling parameter of each feature, used by transformer.
- `transformer::ZScoreTransform`: object for transforming the data during preprocessing.
- `windows::Int`: number of cells in each dimension (i.e., windows=2 results in 4 panels).
- `feat_dim::Int`: dimension of each individual feature, before windowing.
"""
mutable struct PreprocessParameters
    # Affine transformation parameters
    mean::Array{Float64, 1}
    scale::Array{Float64, 1}
    transformer::ZScoreTransform

    # Number of windows
    # NOTE: this is actually number of cells in each dimension
    #   e.g. windows = 2 results in four panels in the 2D image
    windows::Integer

    # Dimension of an individual feature, before considering windowing
    feat_dim::Integer

    # Sigmoid scaling parameter
    sigmoid_scaling::Float64

    # ART module options
    opts_ddvfa::opts_DDVFA

end # PreprocessParameters


"""
    PreprocessParameters(config::ConfParse)

Default intialization for the task detector parameters.

# Arguments
- `config::ConfParse`: global configuration object with fields for preprocessing.
"""
function PreprocessParameters(config::ConfParse)

    # Get the means and scaling parameters
    mean_file = retrieve(config, "params", "mean_file", MEAN_FILE)
    scale_file = retrieve(config, "params", "scale_file", SCALE_FILE)
    # Get the number of windows
    windows = parse(Int64, retrieve(config, "params", "windows", string(WINDOWS)))
    # Get the feature dimension
    feat_dim = parse(Int64, retrieve(config, "params", "feat_dim", string(FEATURE_DIM)))
    sigmoid_scaling = parse(Float64, retrieve(config, "params", "sigmoid_scaling", string(SIGMOID_SCALING)))


    # Load the mean and scale local configs
    local_path = @__DIR__
    mean = vec(readdlm(joinpath(local_path, mean_file), ',', Float64, '\n'))
    scale = vec(readdlm(joinpath(local_path, scale_file), ',', Float64, '\n'))

    # Construct the transformer from the mean and scale parameters
    transformer = ZScoreTransform(windows*feat_dim, 2, mean, scale)

    # Construct the DDVFA options from the taskdet_config.ini with defaults
    opts_ddvfa = opts_DDVFA(
        rho_lb    = parse(Float64,retrieve(config, "ddvfa", "rho_lb", string(DDVFA_RHO_LB))),
        rho_ub    = parse(Float64,retrieve(config, "ddvfa", "rho_ub", string(DDVFA_RHO_UB))),
        alpha     = parse(Float64,retrieve(config, "ddvfa", "alpha", string(DDVFA_ALPHA))),
        beta      = parse(Float64,retrieve(config, "ddvfa", "beta", string(DDVFA_BETA))),
        gamma     = parse(Float64,retrieve(config, "ddvfa", "gamma", string(DDVFA_GAMMA))),
        gamma_ref = parse(Float64,retrieve(config, "ddvfa", "gamma_ref", string(DDVFA_GAMMA_REF))),
        method    = retrieve(config, "ddvfa", "method", DDVFA_METHOD),
        display   = parse(Bool, retrieve(config, "ddvfa", "display", string(DDVFA_DISPLAY)))
    )

    # Construct and return the parameters
    PreprocessParameters(
        mean,
        scale,
        transformer,
        windows,
        feat_dim,
        sigmoid_scaling,
        opts_ddvfa,
    )
end # PreprocessParameters(config::ConfParse)
