# DCCR

Deep Clustering Context Recognition (DCCR); materials for the upcoming paper "Whole-Scene Context Recognition in a Custom AirSim Environment."

## Table of Contents

- [DCCR](#dccr)
  - [Table of Contents](#table-of-contents)
  - [File Structure](#file-structure)

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