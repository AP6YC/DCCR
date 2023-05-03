# Cluster Files

This folder contains scripts and submission files for running experiments on the high-performance computing cluster at MST.
The jobs are mainly `Slurm` submission scripts that simply run one or more of the experiment scripts in `src/experiments`.

## Useful Commands

### Interactive Jobs

This runs an interactive job with one node, many threads, and double the number of default memory per cpu:

```shell
sinteractive --time=12:00:00 --ntasks=32 --nodes=1 --mem-per-cpu=2000
```

This command does the same but on the CUDA nodes:

```shell
sinteractive cuda --time=12:00:00 --gres=gpu:100 --ntasks=32 --nodes=1 --mem-per-cpu=2000
```
