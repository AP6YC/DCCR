"""
    deparallelize.jl

# Description
Removes existing parallel workers.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# Close the workers after simulation
rmprocs(workers())
