```@meta
DocTestSetup = quote
    using OAR, Dates
end
```

```@raw html
<img src="assets/logo.png" width="300">
```

---

These pages serve as the official documentation for the `DCCR` (*Deep Clustering Context Recognition*) project.

## Manual Outline

This documentation is split into the following sections:

```@contents
Pages = [
    "man/guide.md",
    "../examples/index.md",
    "man/contributing.md",
    "man/full-index.md",
    "man/dev-index.md",
]
Depth = 1
```

The [Package Guide](@ref) provides a tutorial to the full usage of the package, while [Examples](@ref examples) gives sample workflows with the various experiments of the project.

The [Contributing](@ref) section outlines how to contribute to the project.
The [Index](@ref main-index) enumerates all public types, functions, and other components with docstrings, whereas internals are listed in the [Developer's Index](@ref dev-main-index).

## About These Docs

Though several different programming languages are used throughout the project, these docs are built around the `Julia` component of the project using the [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) package.

## Documentation Build

This documentation was built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) with the following version and OS:

```@example
using DCCR, Dates # hide
println("DCCR v$(DCCR_VERSION) docs built $(Dates.now()) with Julia $(VERSION) on $(Sys.KERNEL)") # hide
```
