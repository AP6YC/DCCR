# Commands

## Foundry Commands

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

## Git LFS

Remove LFS files and renormalize:

```shell
git add --renormalize .
```

[Git LFS issue](https://github.com/git-lfs/git-lfs/issues/3026#issuecomment-451598434)

