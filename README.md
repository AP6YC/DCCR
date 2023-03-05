# DCCR

Deep Clustering Context Recognition (DCCR); materials for the upcoming ICML paper "Lifelong Context Recognition via Online Deep Clustering."

[issues-url]: https://github.com/AP6YC/DCCR/issues

## Table of Contents

- [DCCR](#dccr)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Installation](#installation)
  - [File Structure](#file-structure)
  - [Contributing](#contributing)
  - [History](#history)
  - [Credits](#credits)
    - [Authors](#authors)

## Usage

Interactive:

```shell
sinteractive --time=03:00:00 --ntasks=16 --nodes=1
sinteractive --time=12:00:00 --ntasks=32 --nodes=1 --mem-per-cpu=2000
```

Interactive CUDA:

```shell
sinteractive -p cuda --time=03:00:00 --gres=gpu:1
sinteractive -p cuda --time=03:00:00 --gres=gpu:1 --ntasks=32 --nodes=1 --mem-per-cpu=2000
```

## Installation

Windows (with Conda):

```shell
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install avalanche-lib[all] jupyterlab ipywidgets julia scikit-learn pandas ipdb tqdm
```

Foundry:

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 avalanche-lib[all] jupyterlab ipywidgets julia scikit-learn pandas ipdb tqdm
```

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

## History

- 6/25/2021 - Initialize the project.
- 4/6/2022 - Create anonymous submission release.

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
