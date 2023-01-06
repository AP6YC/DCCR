"""
    7_unsupervised_mc.jl

Description:
    This script runs an unsupervised learning scenario. After supervised pretraining,
the module learns upon additional data unsupervised and is tested for performance
before and after. This is done as a monte carlo of many simulations.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 6/17/2022: Created and documented.
"""

# Start several processes
using Distributed

# If we parallelize from the command line
if !isempty(ARGS)
    addprocs(parse(Int, ARGS[1]), exeflags="--project=.")
end

# Set the simulation parameters
sim_params = Dict{String, Any}(
    "m" => "ddvfa",
    "seed" => collect(1:1000)
)

@everywhere begin
    # Activate the project in case
    using Pkg
    Pkg.activate(".")

    # Modules
    using Revise            # Editing this file
    using DrWatson          # Project directory functions, etc.

    # Experiment save directory name
    experiment_top = "7_unsupervised_mc"

    # Run the common setup methods (data paths, etc.)
    include(projectdir("src", "setup.jl"))

    # Make a path locally just for the sweep results
    # sweep_results_dir(args...) = results_dir("sweep", args...)
    # sweep_results_dir(args...) = projectdir("work", "data", experiment_top, "sweep", args...)
    sweep_results_dir(args...) = unpacked_dir(experiment_top, "sweep", args...)
    mkpath(sweep_results_dir())

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

    # Create the DDVFA options
    opts = opts_DDVFA(
        gamma = 5.0,
        gamma_ref = 1.0,
        rho_lb = 0.45,
        rho_ub = 0.7,
        method = "single"
    )

    # Sigmoid input scaling
    scaling = 2.0

    # Plotting DPI
    dpi = 350

    # Load the data names and class labels from the selection
    data_dirs, class_labels = get_orbit_names(data_selection)

    # Number of classes
    n_classes = length(data_dirs)

    # Load the orbits
    @info "Worker $(myid()): loading data"
    data = load_orbits(data_dir, data_dirs, scaling)

    combined_data = DataSplitCombined(data)
    # Define a single-parameter function for pmap
    local_sim(dict) = unsupervised_mc(dict, combined_data, opts)

end

# Log the simulation scale
@info "START: $(dict_list_count(sim_params)) simulations across $(nprocs())."

# Turn the dictionary of lists into a list of dictionaries
dicts = dict_list(sim_params)

# Remove impermissible sim options
# filter!(d -> d["rho_ub"] > d["rho_lb"], dicts)
# @info "Testing permutations:" dicts

# Parallel map the sims
pmap(local_sim, dicts)

# Save the data into a binary
pack_data(experiment_top)

println("--- Simulation complete ---")

# Close the workers after simulation
# rmprocs(workers())

# Close the workers after simulation
if !isempty(ARGS)
    rmprocs(workers())
end

# # -----------------------------------------------------------------------------
# # FILE SETUP
# # -----------------------------------------------------------------------------

# # Load Revise for speed and DrWatson for folder pointing
# using Revise            # Editing this file
# using DrWatson          # Project directory functions, etc.

# # Experiment save directory name
# experiment_top = "5_unsupervised_val"

# # Run the common setup methods (data paths, etc.)
# include(projectdir("src", "setup.jl"))

# # -----------------------------------------------------------------------------
# # OPTIONS
# # -----------------------------------------------------------------------------

# # Saving names
# plot_name = "5_accuracy.png"
# heatmap_name = "5_heatmap.png"

# # n_F2_name = "1_n_F2.tex"
# n_cat_name = "5_n_cat.tex"

# # Select which data entries to use for the experiment
# data_selection = [
#     "dot_dusk",
#     "dot_morning",
#     # "emahigh_dusk",
#     # "emahigh_morning",
#     "emalow_dusk",
#     "emalow_morning",
#     "pr_dusk",
#     "pr_morning",
# ]

# # Create the DDVFA options
# opts = opts_DDVFA(
#     gamma = 5.0,
#     gamma_ref = 1.0,
#     rho_lb = 0.45,
#     rho_ub = 0.7,
#     method = "single"
# )

# # Sigmoid input scaling
# scaling = 2.0

# # -----------------------------------------------------------------------------
# # EXPERIMENT SETUP
# # -----------------------------------------------------------------------------

# # Load the data names and class labels from the selection
# data_dirs, class_labels = get_orbit_names(data_selection)

# # Number of classes
# n_classes = length(data_dirs)

# # Load the data
# data = load_orbits(data_dir, scaling)

# # Create the DDVFA module and set the data config
# ddvfa = DDVFA(opts)
# ddvfa.config = DataConfig(0, 1, 128)

# # -----------------------------------------------------------------------------
# # TRAIN/TEST
# # -----------------------------------------------------------------------------

# # Train in batch
# y_hat_train = train!(ddvfa, data.train_x, y=data.train_y)
# y_hat = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

# # Calculate performance on training data, testing data, and with get_bmu
# perf_train = performance(y_hat_train, data.train_y)
# perf_test = performance(y_hat, data.test_y)

# # Format each performance number for comparison
# @printf "Batch training performance: %.4f\n" perf_train
# @printf "Batch testing performance: %.4f\n" perf_test

# # -----------------------------------------------------------------------------
# # UNSUPERVISED TRAIN/TEST ON VALIDATION DATA
# # -----------------------------------------------------------------------------

# # Train in batch, unsupervised
# # y_hat_train_val = train!(ddvfa, data.val_x, y=data.val_y)
# y_hat_train_val = train!(ddvfa, data.val_x)
# # If the category is not in 1:6, replace the label as 7 for the new/incorrect bin
# replace!(x -> !(x in collect(1:n_classes)) ? 7 : x, ddvfa.labels)
# y_hat_val = AdaptiveResonance.classify(ddvfa, data.test_x, get_bmu=true)

# # Calculate performance on training data, testing data, and with get_bmu
# perf_train_val = performance(y_hat_train_val, data.val_y)
# perf_test_val = performance(y_hat, data.test_y)

# # Format each performance number for comparison
# @printf "Batch training performance: %.4f\n" perf_train_val
# @printf "Batch testing performance: %.4f\n" perf_test_val

# # -----------------------------------------------------------------------------
# # ACCURACY PLOTTING
# # -----------------------------------------------------------------------------

# # Create an accuracy grouped bar chart
# p = create_comparison_groupedbar(data, y_hat_val, y_hat, class_labels, percentages=true, extended=true)
# display(p)

# # Save the plot
# savefig(p, results_dir(plot_name))
# savefig(p, paper_results_dir(plot_name))

# # -----------------------------------------------------------------------------
# # CATEGORY ANALYSIS
# # -----------------------------------------------------------------------------

# # Save the number of F2 nodes and total categories per class
# n_F2, n_categories = get_n_categories(ddvfa)
# @info "F2 nodes:" n_F2
# @info "Total categories:" n_categories

# # Create a LaTeX table from the categories
# df = DataFrame(F2 = n_F2, Total = n_categories)
# table = latexify(df, env=:table)

# # Save the categories table to both the local and paper results directories
# open(results_dir(n_cat_name), "w") do io
#     write(io, table)
# end
# open(paper_results_dir(n_cat_name), "w") do io
#     write(io, table)
# end

# # -----------------------------------------------------------------------------
# # CONFUSION HEATMAP
# # -----------------------------------------------------------------------------

# # Normalized confusion heatmap
# # norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
# h = create_confusion_heatmap(class_labels, data.test_y, y_hat)
# display(h)

# # Save the heatmap to both the local and paper results directories
# savefig(h, results_dir(heatmap_name))
# savefig(h, paper_results_dir(heatmap_name))
