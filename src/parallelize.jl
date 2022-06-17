"""
    parallelize.jl

Adds a number of workers for parallel processing.
Currently, the number of workers is passed directly to addprocs.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 1/15/2022: Created.
- 2/17/2022: Documented.
"""

# Start several processes
using Distributed
addprocs(26, exeflags="--project=.")
# addprocs(10, exeflags="--project=.")
