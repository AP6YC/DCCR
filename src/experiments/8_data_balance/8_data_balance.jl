"""
    8_balance.jl

Description:
    This script produces the Shannon entropy of the dataset to quantify the
dataset balance/imbalance.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 7/29/2022: Created and documented.
"""

# -----------------------------------------------------------------------------
# FILE SETUP
# -----------------------------------------------------------------------------

using Revise
using DrWatson

# Experiment save directory name
experiment_top = "8_data_balance"

# Run the common setup methods (data paths, etc.)
include(projectdir("src", "setup.jl"))

# -----------------------------------------------------------------------------
# SCRIPT
# -----------------------------------------------------------------------------

# Manually entered sizes of the data
data_sizes = Dict(
    "DOTD" => Dict(
        "train" => 337,
        "test" => 348,
    ),
    "DOTM" => Dict(
        "train" => 349,
        "test" => 321,
    ),
    "EMAD" => Dict(
        "train" => 296,
        "test" => 279,
    ),
    "EMAM" => Dict(
        "train" => 270,
        "test" => 276,
    ),
    "PRD" => Dict(
        "train" => 280,
        "test" => 309,
    ),
    "PRM" => Dict(
        "train" => 281,
        "test" => 283,
        # "test" => 2, # for testing entropy
    ),
    # "" => Dict(
    #     "train" =>,
    #     "test" =>,
    # ),
)

# Get the data vectors and other parameters
train_vec = [data_sizes[key]["train"] for (key, _) in data_sizes]
test_vec = [data_sizes[key]["test"] for (key, _) in data_sizes]
n_classes = length(train_vec)
n_train = sum(train_vec)
n_test = sum(test_vec)

# OLD

H_train = -sum(train_vec./n_train .* log.(train_vec/n_train))
H_test = -sum(test_vec./n_test .* log.(test_vec/n_test))

B_train = H_train / log(n_classes)
B_test = H_test / log(n_classes)

@info "Balances:" B_train B_test

# Shannon diversity index
H_SDI_train = (n_train * log(n_train) - sum(train_vec .* log.(train_vec))) / n_train
H_SDI_test = (n_test * log(n_test) - sum(test_vec .* log.(test_vec))) / n_test

# Shannon equitability index
H_SEI_train = H_SDI_train / log(n_classes)
H_SEI_test = H_SDI_test / log(n_classes)

@info "Shannon Diversity Indices and Equitability Indices:" H_SDI_train H_SDI_test H_SEI_train H_SEI_test
