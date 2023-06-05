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
# DOWNLOAD LARGE ASSETS
# -----------------------------------------------------------------------------

# # Point to the raw FileStorage location on GitHub
# top_url = raw"https://media.githubusercontent.com/media/AP6YC/FileStorage/main/AdaptiveResonance/"
# # List all of the files that we need to use in the docs
# files = [
#     "header.png",
#     "art.png",
#     "artmap.png",
#     "ddvfa.png",
# ]
# # Make a destination for the files
# download_folder = joinpath("src", "assets", "downloads")
# mkpath(download_folder)
# download_list = []
# # Download the files one at a time
# for file in files
#     # Point to the correct file that we wish to download
#     src_file = top_url * file * "?raw=true"
#     # Point to the correct local destination file to download to
#     dest_file = joinpath(download_folder, file)
#     # Add the file to the list that we will append to assets
#     push!(download_list, dest_file)
#     # If the file isn't already here, download it
#     if !isfile(dest_file)
#         download(src_file, dest_file)
#     end
# end

# -----------------------------------------------------------------------------
# GENERATE DOCUMENTATION
# -----------------------------------------------------------------------------

assets = [
    joinpath("assets", "favicon.ico"),
]

makedocs(
    modules = [DCCR],
    sitename = "DCCR",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = assets,
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Guide" => "man/guide.md",
        ],
        "Internals" => [
            "Index" => "man/full-index.md",
            "Dev Index" => "man/dev-index.md",
            "Contributing" => "man/contributing.md",
        ],
    ],
    repo="https://github.com/AP6YC/DCCR/blob/{commit}{path}#L{line}",
    authors="Sasha Petrenko",
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
