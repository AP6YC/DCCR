"""
    3_analyze.jl

Description:
    This script takes the results of the Monte Carlo of shuffled simulations
and generates plots of their statistics.

Author: Sasha Petrenko <sap625@mst.edu>
Date: 1/17/2022
"""

using Revise
using DataFrames
# using Plots
# using PlotThemes
using DrWatson

# Plotting style
# pyplot()
# theme(:dark)

sweep_dir = projectdir("work", "results", "3_shuffled_mc", "sweep")
# df = collect_results(projectdir("work/results/sweep/"))
# df = collect_results("work\\results\\sweep\\2021-04-19_16-11-28")
df = collect_results(sweep_dir)

# df_high = filter(row -> row[:test_perf] > 0.97, df)
# df_high[:, Not([:method, :arch, :path])]




# df_high = filter(row -> row[:test_perf] > 0.94 && row[:train_perf] > 0.94, df)

# plot
# xlabel!("F2 Category")
# ylabel!("# Weights")
# title!(string(cell)*(cell > 1 ? " Windows" : " Window"))
# # fig_path = joinpath(cell, string(cell))
# fig_dir = joinpath("work/results", subdir)
# mkpath(fig_dir)
# fig_path = joinpath(fig_dir, string(cell))
# savefig(fig_path)
