# Package Guide

To work with the `DCCR` project, you should know:

- [How to install the package](@ref installation)

## [Installation](@id installation)

Because it is an experimental research repository, the DCCR package is not registered on JuliaHub.
To set `Julia` component the project up, you must:

- [Download a `Julia`](https://julialang.org/downloads/) distribution and install it on your system
- [Git clone this repository](https://github.com/AP6YC/DCCR) or download a [zip](https://github.com/AP6YC/DCCR/archive/refs/heads/main.zip).
- Run `julia` within the top of the `DCCR` directory, and run the following commands to instantiate the package:

```julia-repl
julia> ]
(@v1.8) pkg> activate .
(DCCR) pkg> instantiate
```

This will download all of the dependencies of the project and precompile where possible.

!!! note "Note"
    This project is still under development, so detailed usage guides beyond installation have not yet been written about the package's functionality.
    Please see the other sections of this documentation for examples, definition indices, and more.
