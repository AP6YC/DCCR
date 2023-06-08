"""
    DCCR.jl

# Description
Definition of the `DCCR` module, which encapsulates experiment driver code.
"""

"""
A module encapsulating the experiment driver code for the `DCCR` project.

# Imports

The following names are imported by the package as dependencies:
$(IMPORTS)

# Exports

The following names are exported and available when `using` the package:
$(EXPORTS)
"""
module DCCR

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Usings/imports for the whole package declared once
using Reexport              # Reexport submodule exports

# Reexport tools used in experiment scripts
@reexport using AdaptiveResonance   # ART algorithms, DDVFA, FuzzyART, etc.
@reexport using Printf              # Formatted number printing
@reexport using DataFrames          # DataFrame
@reexport using Latexify            # latexify
@reexport using Random              # Random subsequence

# Full usings (which supports comma-separated import notation)
using
    DataStructures,         # Dequeue
    DelimitedFiles,
    DocStringExtensions,    # Docstring utilities
    DrWatson,               # Scientific project commands
    JSON                    # JSON file load/save
    Logging,                # Printing diagnostics
    MLBase,                 # confusmat
    MLDataUtils,            # stratifiedobs
    NumericalTypeAliases,   # RealMatrix, IntegerVector, etc.
    Plots,                  # Plotting
    ProgressMeter,          # Progress bars
    PyCall,                 # PyObject
    StatsBase,
    StatsPlots              # groupedbar

# using HDF5              # Loading .h5 activation files

import Tar

# Precompile concrete type methods
using PrecompileSignatures: @precompile_signatures

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

# Necessary to download data without prompts to custom folders
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Include all files
include("version.jl")           # Exported constant for the version of the project
include("lib_c3/lib_c3.jl")     # C3 library code
include("lib_l2.jl")            # Lifelong learning library code

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

# Export all public names
export DCCR_VERSION

# -----------------------------------------------------------------------------
# PRECOMPILE
# -----------------------------------------------------------------------------

# Precompile any concrete-type function signatures
@precompile_signatures(DCCR)

end
