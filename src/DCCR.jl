module DCCR

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Usings/imports for the whole package declared once

# Full usings (which supports comma-separated import notation)
using
    AdaptiveResonance,
    DocStringExtensions,    # Docstring utilities
    DrWatson,
    NumericalTypeAliases,
    Reexport

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
include("version.jl")   # Exported constant for the version of the project

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
