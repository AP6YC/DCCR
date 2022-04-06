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
├── opts                    // Options: files for each experiment, learner, etc.
├── work                    // Work: Temporary file location (weights, datasets)
│   ├── data                //      Datasets
│   └── models              //      Model weights
├── .gitattributes          // Git: definitions for LFS patterns
├── .gitignore              // Git: .gitignore for the whole project
├── README.md               // Doc: this document
```

## Contributing

Please raise an [issue][issues-url].

## History

- 6/25/2021 - Initialize the project.
