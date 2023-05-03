"""
    cluster_parallelize.jl

# Description
Adds a number of workers for parallel processing on the MST cluster.
Currently, the number of workers is passed directly to addprocs from the first script argument.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# Start several processes
using Distributed
addprocs(ARGS[1], exeflags="--project=.")
