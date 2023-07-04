"""
    parallelize.jl

# Description
Adds a number of workers for parallel processing.
Currently, the number of workers is passed directly to addprocs.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# Start several processes
using Distributed
# addprocs(28, exeflags="--project=.")
addprocs(24, exeflags="--project=.")
# addprocs(10, exeflags="--project=.")
