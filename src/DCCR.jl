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

# Full usings (which supports comma-separated import notation)
using
    AdaptiveResonance,      # ART algorithms, DDVFA, FuzzyART, etc.
    ArgParse,               # ArgParseSettings
    DataFrames,             # DataFrame
    DataStructures,         # Dequeue
    Dates,                  # Dates.format
    DelimitedFiles,
    DocStringExtensions,    # Docstring utilities
    DrWatson,               # Scientific project commands
    JLD2,                   # JLD2.load
    JSON,                   # JSON file load/save
    # Latexify,               # latexify
    Logging,                # Printing diagnostics
    MLBase,                 # confusmat
    MLDataUtils,            # stratifiedobs
    NumericalTypeAliases,   # RealMatrix, IntegerVector, etc.
    Plots,                  # Plotting
    Printf,                 # Formatted number printing
    ProgressMeter,          # Progress bars
    PyCall,                 # PyObject
    Random,                 # Random subsequence
    StatsPlots              # groupedbar

# using HDF5              # Loading .h5 activation files

import
    Tar,
    YAML

# Precompile concrete type methods
using PrecompileSignatures: @precompile_signatures

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

# Necessary to download data without prompts to custom folders
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
# Suppress display on headless systems
ENV["GKSwstype"] = 100

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Aggregate of all library code
include("lib/lib.jl")

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
