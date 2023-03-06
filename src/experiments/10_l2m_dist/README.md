# 10_l2m_dist

This experiment runs a condensed L2M scenario in Julia with the l2logger and l2metrics Python packages.

The files in this experiment are run in numerical order:

1. `1_setup_l2_packages.jl`: this script sets up the l2metrics and l2logger packages in the Python environment set in `src/setup_l2.jl`.
2. `2_gen_scenarios.jl`: generates a scenario in reverse from the full dataset.
3. `3_dist_driver.jl`: runs the full L2 scenario with the DDVFA agent on the generated scenario from the previous script, logging with the `l2logger`.
4. `4_dist_metrics_par.jl`: generates the l2 metrics with the `l2metrics` package in parallel for speed.
The serial version of this script is implemented in `4_dist_metrics.jl`.
5. `5_dist_l2m_table.jl`: loads and aggregates all of the desired l2metrics from the previous script and puts it into one .csv file.
6. `6_stats.jl`: loads the .csv file of l2 stats from the previous script and generates a LaTeX table of statistics for the l2 metrics, saving to the paper directory.
