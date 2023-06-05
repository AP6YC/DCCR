"""
    ddvfa.py

# Description
This file contains an old definition of an Avalanche `strategy` using DDVFA.

This is a bit of an abuse of the framework, but it is done because creating a
DDVFA `model` requires creating an entirely new `strategy` that doesn't assume
the use of an optimizer and subsequent gradient-based training, which will be
done in due time.
"""

import julia
julia.install()

import time

from torch.utils.data import DataLoader
from julia import Main as jl
from sklearn.metrics import accuracy_score
import ipdb
import torch
# from torchvision.transforms import Lambda
from tqdm import tqdm

from torchvision.models import resnet50, resnet18
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.metrics import accuracy_score
from statistics import mean


class DDVFAStrategy():
    """DDVFA Strategy"""

    # def __init__(self, preprocessed=False):
    def __init__(self, projectdir):
        # jl.project_dir = projectdir()
        jl.project_dir = projectdir
        jl.eval("using Pkg; Pkg.activate(project_dir)")
        jl.eval("using AdaptiveResonance")
        jl.eval("art = DDVFA(rho_lb=0.4, rho_ub=0.75)")
        # jl.eval("art = DDVFA(rho_lb=0.5, rho_ub=0.75)")
        jl.eval("art.config = DataConfig(0, 1.0, 49)")
        # jl.eval("art.config = DataConfig(0, 1.0, 196)")

    def jl_train(self):
        jl.eval("y_hats = train!(art, features, y=labels)")

    def train(self, experience):
        train_dataset = experience.dataset
        t = experience.task_label
        train_data_loader = DataLoader(
            dataset=train_dataset,
            # pin_memory=True,
            # batch_size=90,
            batch_size=256,
        )

        print(experience.dataset.__len__())
        # for mb in tqdm(train_data_loader):

        jl_eval = 0.0
        perfs = []
        pbar = tqdm(train_data_loader)
        for mb in pbar:
            data, labels, tasks = mb
            # jl.features = self.ext_features(data)
            jl.features = data.squeeze().numpy().transpose()
            jl.labels = labels.numpy() + 1
            # ipdb.set_trace()

            start_time = time.time()
            self.jl_train()
            end_time = time.time()
            jl_eval = end_time - start_time
            pbar.set_postfix({"jl time": jl_eval})

            y_hats = jl.y_hats - 1
            perfs.append(accuracy_score(labels, y_hats))

        return mean(perfs)

    def jl_eval(self):
        if jl.eval("length(unique(art.labels))") == 1:
            jl.y_hats = jl.eval("unique(art.labels)").repeat(len(jl.labels))
        else:
            jl.eval("y_hats = classify(art, features, get_bmu=true)")

    def eval(self, experience):
        eval_dataset = experience.dataset
        t = experience.task_label

        eval_data_loader = DataLoader(
            dataset=eval_dataset,
            pin_memory=True,
            # batch_size=90,
            batch_size=256,
        )

        print(experience.dataset.__len__())

        jl_eval = 0.0
        perfs = []
        pbar = tqdm(eval_data_loader)
        for mb in pbar:
            data, labels, tasks = mb
            # jl.features = self.ext_features(data)

            jl.features = data.squeeze().numpy().transpose()
            jl.labels = labels.numpy() + 1
            start_time = time.time()
            # jl.eval("y_hats = classify(art, features, get_bmu=true)")
            self.jl_eval()
            end_time = time.time()
            jl_eval = end_time - start_time
            pbar.set_postfix({"jl time": jl_eval})

            y_hats = jl.y_hats - 1
            perfs.append(accuracy_score(labels, y_hats))

        return mean(perfs)

# print_allocated_memory()