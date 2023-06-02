using Documenter
using DCCR

makedocs(
    sitename = "DCCR",
    format = Documenter.HTML(),
    modules = [DCCR]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
