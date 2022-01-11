# DCCR

Deep Clustering Context Recognition (DCCR); materials for the upcoming paper "Whole-Scene Context Recognition in a Custom AirSim Environment."

[issues-url]: https://github.com/AP6YC/DCCR/issues

## Table of Contents

- [DCCR](#dccr)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [File Structure](#file-structure)
  - [Contributing](#contributing)
  - [History](#history)
  - [Credits](#credits)
    - [Authors](#authors)
    - [Software](#software)

## Usage

TODO

## File Structure

```
DCCR
├── dockerfiles             // Dockerfiles: for deployment
├── julia                   // Source: julia scripts and modules
│   ├── experiments         //      Experiment scripts
│   ├── lib                 //      Common experimental code
│   ├── meta-icvi           //      Meta-ICVI development scripts and libs
│   └── utils               //      Utility scripts (data inspection, etc.)
├── python                  // Source: python scripts and modules
├── cluster                 // Cluster: submission scripts for the MST HPC cluster
│   ├── scripts             //      Utility scripts
│   └── sub                 //      Submission scripts
├── opts                    // Options: files for each experiment, learner, etc.
├── test                    // Test: Pytest unit, integration, and environment tests
├── work                    // Work: Temporary file location (weights, datasets)
│   ├── data                //      Datasets
│   └── models              //      Model weights
├── .gitattributes          // Git: definitions for LFS patterns
├── .gitignore              // Git: .gitignore for the whole project
├── README.md               // Doc: this document
└── requirements.txt        // Doc: pip requirements file for standard venv
```

## Contributing

Please raise an [issue][issues-url].

## History

- 6/25/2021 - Initialize the project.
- 8/23/2021 - Refactor L2MTaskDetector as a Julia dependency.

## Credits

### Authors

- Sasha Petrenko <sap625@mst.edu>
- Andrew Brna <>

### Software

- https://github.com/AP6YC/L2MTaskDetector.jl
