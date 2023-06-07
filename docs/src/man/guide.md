# Package Guide

To work with the `DCCR` project, you should know:

- [How to install the project](@ref guide-installation)
- [An overview of the project](@ref guide-overview)
- [How to run experiments](@ref guide-experiments)

## [Installation](@id guide-installation)

Because it is an experimental research repository, the `DCCR` package is not registered on JuliaHub.
To set `Julia` component the project up, you must:

- [Download a `Julia`](https://julialang.org/downloads/) distribution and install it on your system
- [Git clone this repository](https://github.com/AP6YC/DCCR) or download a [zip](https://github.com/AP6YC/DCCR/archive/refs/heads/main.zip).
- Run `julia` within the top of the `DCCR` directory, and run the following commands to instantiate the package:

```julia-repl
julia> ]
(@v1.9) pkg> activate .
(DCCR) pkg> instantiate
```

This will download all of the dependencies of the project and precompile where possible.

## [Overview](@id guide-overview)

The `DCCR` project is mainly a [`Julia`](https://julialang.org/) programming language research project, so it is not designed as a package for use and installation through [`JuliaHub`](https://juliahub.com/ui/Packages) as other packages are.
However, the main driver libraries of the project are bundled into a `DCCR` module, so this module is loaded in the preamble of all experiments as a concise way of loading common code.

The project utilizes [`DrWatson`](https://juliadynamics.github.io/DrWatson.jl/dev/) for workflow utilities such as directory operations, results saving/loading, and simulation configurations.
The file structure of this project differs slightly from the `DrWatson` [default setup](https://juliadynamics.github.io/DrWatson.jl/dev/project/l), so extra utilities are used for pointing to the correct source data directory and destination results directory according to each experiment.

In addition to the `Julia` components, some experiments are written in [`Python`](https://www.python.org/), particularly those that utilize the [`avalanche`](https://avalanche.continualai.org/) library for continual learning experiments.
Some of these even utilize `PyCall` to run `Julia` code from `Python` and vice versa.

Each experiment contains a `README` outlining the setup and usage of the experiment such as in the setup of custom `PyCall` `Python` environments and in the running of parallel and distibuted experiments.

## [Experiments](@id guide-experiments)

To run an experiment, [setup the `DCCR` project](@ref guide-installation) on your target system and run the experiment either in an interactive session with `include(...)`:

```julia
include("src/experiments/1_accuracy/1_unshuffled.jl")
```

or from a terminal command line (from the top of the project directory):

```sh
julia --project=. src/experiments/1_accuracy/1_unshuffled.jl
```

!!! note "Note"
    This project is still under development, so detailed usage guides beyond this have not yet been written about the project's functionality.
    Please see the other sections of this documentation for examples, definition indices, and more.
