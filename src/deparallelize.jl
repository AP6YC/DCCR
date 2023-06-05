"""
    deparallelize.jl

Removes existing parallel workers.

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 1/15/2022: Created.
- 2/17/2022: Documented.
"""

# Close the workers after simulation
rmprocs(workers())
