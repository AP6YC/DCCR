"""
    make.jl

# Description
This file builds the documentation for the DCCR project
using Documenter.jl and other tools.
"""
# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    Documenter,
    Pkg

# using DCCR

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Fix GR headless errors
ENV["GKSwstype"] = "100"

# Get the current workind directory's base name
current_dir = basename(pwd())
@info "Current directory is $(current_dir)"

# If using the CI method `julia --project=docs/ docs/make.jl`
#   or `julia --startup-file=no --project=docs/ docs/make.jl`
if occursin("DCCR", current_dir)
    push!(LOAD_PATH, "../src/")
# Otherwise, we are already in the docs project and need to dev the above package
elseif occursin("docs", current_dir)
    Pkg.develop(path="..")
# Otherwise, building docs from the wrong path
else
    error("Unrecognized docs setup path")
end

# Inlude the local package
using DCCR

# using JSON
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

# -----------------------------------------------------------------------------
# GENERATE DOCUMENTATION
# -----------------------------------------------------------------------------

makedocs(
    sitename = "DCCR",
    format = Documenter.HTML(),
    modules = [DCCR]
)

# -----------------------------------------------------------------------------
# DEPLOY
# -----------------------------------------------------------------------------

deploydocs(
    repo="github.com/AP6YC/DCCR.git",
    # devbranch="develop",
    devbranch="main",
    # push_preview = should_push_preview(),
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
