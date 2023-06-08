# DCCR

[![dccr-header](docs/src/assets/logo.png)][docs-url]

Deep Clustering Context Recognition (DCCR); materials for the upcoming TNNLS paper "Lifelong Context Recognition via Online Deep Feature Clustering."
Please see the [documentation][docs-url].

| **Documentation** | **Docs Build Status**|  **Testing Status** |
|:-----------------:|:--------------------:|:-------------------:|
| [![Docs][docs-img]][docs-url] | [![Docs Status][doc-status-img]][doc-status-url] | [![CI Status][ci-img]][ci-url] |
| **Coveralls** | **Codecov** | **Zenodo DOI** |
 [![Coveralls][coveralls-img]][coveralls-url] | [![Codecov][codecov-img]][codecov-url] | [![Zenodo DOI][doi-img]][doi-url] |


[doc-status-img]: https://github.com/AP6YC/DCCR/actions/workflows/Documentation.yml/badge.svg
[doc-status-url]: https://github.com/AP6YC/DCCR/actions/workflows/Documentation.yml

[docs-img]: https://img.shields.io/badge/docs-blue.svg
[docs-url]: https://AP6YC.github.io/DCCR/dev/

[ci-img]: https://github.com/AP6YC/DCCR/workflows/CI/badge.svg
[ci-url]: https://github.com/AP6YC/DCCR/actions?query=workflow%3ACI

[codecov-img]: https://codecov.io/gh/AP6YC/DCCR/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AP6YC/DCCR

[coveralls-img]: https://coveralls.io/repos/github/AP6YC/DCCR/badge.svg?branch=main
[coveralls-url]: https://coveralls.io/github/AP6YC/DCCR?branch=main

[issues-url]: https://github.com/AP6YC/DCCR/issues

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.8017807.svg
[doi-url]: https://doi.org/10.5281/zenodo.8017807

## Table of Contents

- [DCCR](#dccr)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [File Structure](#file-structure)
  - [Contributing](#contributing)
  - [Attribution](#attribution)
    - [Authors](#authors)
    - [License](#license)
    - [Useful Links](#useful-links)
    - [Assets](#assets)
    - [Citation](#citation)

## Usage

Experiments are enumerated in `src/experiments`.
Each has a `README.md` that describes the experiment and how to run it.
Most experiments only require instantiating the Julia project in this repo with

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

and running the script in the experiment folder with either the shell command:

```shell
julia src/experiments/1_accuracy/1_unshuffled.jl
```

or in an existing REPL environment with the include command:

```julia
include("src/experiments/1_accuracy/1_unshuffled.jl")
```

Experiments with multiple stages or multiple interpreters (Julia, Python, and shell script) contain details for their reproducibilty.

## File Structure

An explanation of the `DCCR` project file structure can be found [in the hosted documentation][package-structure].

[package-structure]: https://ap6yc.github.io/DCCR/dev/man/contributing/#Package-Structure

## Contributing

If you have an issue with the project, please raise an [issue][issues-url].
If you would instead like to contribute to the package, please see the [contributing guide][contributing-guide].

[contributing-guide]: https://ap6yc.github.io/DCCR/dev/man/contributing/

## Attribution

### Authors

- Sasha Petrenko <sap625@mst.edu>
- Andrew Brna <andrew.brna@teledyne.com>

### License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

### Useful Links

The following resources are referenced in this project or are useful resources for reference:

- [Avalanche docs](https://avalanche.continualai.org/)
- [Avalanche continual-learning-baselines repo](https://github.com/ContinualAI/continual-learning-baselines)
- Deep Streaming Linear Discriminant Analysis (DSLDA):
  - [CVPR open access paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf.)
  - [GitHub repo](https://github.com/tyler-hayes/Deep_SLDA)
- Continual Prototype Evolution (CoPE):
  - [CVPR open access paper](https://openaccess.thecvf.com/content/ICCV2021/papers/De_Lange_Continual_Prototype_Evolution_Learning_Online_From_Non-Stationary_Data_Streams_ICCV_2021_paper.pdf)
  - [GitHub repo](https://github.com/Mattdl/ContinualPrototypeEvolution)
  - [arXiv](https://arxiv.org/abs/2009.00919)

### Assets

The following external assets are used in this project by attribution:

- [Drone-case icons created by Smashicons](https://www.flaticon.com/free-icons/drone-case) ([drone_2738988](https://www.flaticon.com/free-icon/drone_2738988))

### Citation

This project has a [citation file](CITATION.cff) file that generates citation information for the package and corresponding JOSS paper, which can be accessed at the "Cite this repository button" under the "About" section of the GitHub page.

You may also cite this repository with the following BibTeX entry:

```bibtex
@software{Petrenko_AP6YC_DCCR_2023,
  author = {Petrenko, Sasha},
  doi = {10.5281/zenodo.8017806},
  month = jun,
  title = {{AP6YC/DCCR}},
  year = {2023}
}
```
