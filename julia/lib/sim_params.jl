using DrWatson

function get_sim_params()

# Simulation parameters
sim_params = Dict{String,Any}(
    # "cell" => [1, 2, 3, 4],
    "cell" => [1, 2],
    "use_alt" => [true, false],
    "arch" => "ddvfa",
    "method" => "weighted",
    # "gamma" => [5],
    "gamma" => Array(LinRange(3, 5, 10)),
    # "rho_ub" => [0.7],
    "rho_ub" => Array(LinRange(0.5, 0.9, 10)),
    # "rho_lb" => [0.6],
    "rho_lb" => Array(LinRange(0.4, 0.8, 10)),
    # "rho" => [0.6]
)

# Turn the dictionary of lists into a list of dictionaries
dicts = dict_list(sim_params)

# Remove impermissible sim options
filter!(d -> d["rho_ub"] > d["rho_lb"], dicts)
@info "Testing permutations:" dicts