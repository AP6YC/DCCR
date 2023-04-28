# DCCR

Deep Clustering Context Recognition (DCCR); materials for the upcoming ICML paper "Lifelong Context Recognition via Online Deep Clustering."

[issues-url]: https://github.com/AP6YC/DCCR/issues

## Table of Contents

- [DCCR](#dccr)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [File Structure](#file-structure)
  - [Contributing](#contributing)
  - [Credits](#credits)
    - [Authors](#authors)
    - [License](#license)
    - [Useful Citation Links](#useful-citation-links)

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

```
DCCR
├── dockerfiles             // Dockerfiles: for deployment
├── src                     // Source: julia scripts and modules
│   ├── experiments         //      Experiment scripts
│   ├── lib                 //      Common experimental code
│   └── utils               //      Utility scripts (data inspection, etc.)
├── opts                    // Options: files for each experiment, learner, etc.
├── test                    // Test: Pytest unit, integration, and environment tests
├── work                    // Work: Temporary file location (weights, datasets)
│   ├── data                //      Datasets
│   ├── models              //      Model weights
│   └── results             //      Generated results
├── .gitattributes          // Git: definitions for LFS patterns
├── .gitignore              // Git: .gitignore for the whole project
├── LICENSE                 // Git: license for the project
├── Project.toml            // Julia: project dependencies
└── README.md               // Doc: this document
```

## Contributing

Please raise an [issue][issues-url].

## Credits

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

### Useful Citation Links

- [Avalanche docs](https://avalanche.continualai.org/)
- [Avalanche continual-learning-baselines repo](https://github.com/ContinualAI/continual-learning-baselines)
- Deep Streaming Linear Discriminant Analysis (DSLDA):
  - [CVPR open access paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf.)
  - [GitHub repo](https://github.com/tyler-hayes/Deep_SLDA)
- Continual Prototype Evolution (CoPE):
  - [CVPR open access paper](https://openaccess.thecvf.com/content/ICCV2021/papers/De_Lange_Continual_Prototype_Evolution_Learning_Online_From_Non-Stationary_Data_Streams_ICCV_2021_paper.pdf)
  - [GitHub repo](https://github.com/Mattdl/ContinualPrototypeEvolution)
  - [arXiv](https://arxiv.org/abs/2009.00919)