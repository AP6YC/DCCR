"""
    ddvfa_foundry.py

# Description

Definition of the DDVFA module for the linux cluster running these experiments.
This is a separate file from ddvfa.py due to pyjulia problems on the cluster.
This file defines an Avalanche `strategy` using DDVFA.
This is a bit of an abuse of the framework, but it is done because creating a
DDVFA `model` requires creating an entirely new `strategy` that doesn't assume
the use of an optimizer and subsequent gradient-based training, which will be
done in due time.
"""

# Good ol' pyjulia problems:
# https://pyjulia.readthedocs.io/en/latest/troubleshooting.html#your-python-interpreter-is-statically-linked-to-libpython
# Pointing to the runtime and precompiling PyCall in this namespace because
# it simply refuses to work when inside a function or class.

# Import and setup the Julia runtime
import julia
runtime = "/home/sap625/julia"
julia.install(julia=runtime)
jl_run = julia.Julia(
    runtime=runtime,
    compiled_modules=False,
)

# Point to the Main namespace for all eval calls
from julia import Main

import time

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
# import ipdb
# import torch
from tqdm import tqdm

# from torchvision.models import resnet50, resnet18
# from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision import models
# from torchvision.models.feature_extraction import create_feature_extractor

from statistics import mean

from typing import Union
from pathlib import Path

# from utils import print_allocated_memory


class DDVFAStrategy():
    """DDVFAStrategy : a DDVFA module implemented as an Avalanche strategy.

    This should really be a model, but that would require creating an Avalanche
    strategy that doesn't use an optimizer (left as an exercise for the reader).
    """

    # def __init__(self, preprocessed=False):
    def __init__(
        self,
        projectdir: Union[str, Path],
        # runtime="julia",
    ):
        """Creates a DDVFA Avalanche strategy.

        Requires the directory containing the pre-instantiated Julia project.

        Parameters
        ----------
        projectdir : Union[str, Path]
            Directory containing the pre-instantiated Julia project directory.
        """
        # SETTING UP THE RUNTIME INSIDE THE CLASS DOESN'T WORK FOR SOME REASON
        # I'VE SPENT TOO LONG DEBUGGING THIS AND NOT ENOUGH TIME ON MY DISSERTATION
        # # Install PyCall?
        # julia.install(julia=runtime)

        # # Setup the runtime
        # self.jl_run = julia.Julia(
        #     runtime=runtime,
        #     compiled_modules=False,
        # )

        # Point to the actual global namespace
        self.jl = Main

        # Setup/activate the project
        self.jl.project_dir = projectdir
        self.jl.eval("using Pkg; Pkg.activate(project_dir)")
        # Setup the ART module
        self.jl.eval("using AdaptiveResonance")
        self.jl.eval("art = DDVFA(rho_lb=0.4, rho_ub=0.75)")
        # jl.eval("art = DDVFA(rho_lb=0.5, rho_ub=0.75)")
        self.jl.eval("art.config = DataConfig(0, 1.0, 49)")
        # self.jl.eval("art.config = DataConfig(0, 1.0, 196)")

    def jl_train(self):
        self.jl.eval("y_hats = train!(art, features, y=labels)")

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
            # Extract the elements of the minibatch
            data, labels, tasks = mb
            # Load the features and labels into the Julia workspace
            self.jl.features = data.squeeze().numpy().transpose()
            self.jl.labels = labels.numpy() + 1     # Julia's 1-indexing

            # Training with timing
            start_time = time.time()
            self.jl_train()
            end_time = time.time()
            jl_eval = end_time - start_time
            pbar.set_postfix({"jl time": jl_eval})

            # Extract the training classification estimates
            y_hats = self.jl.y_hats - 1
            # Append the accuracy for the minibatch
            perfs.append(accuracy_score(labels, y_hats))

        # Return the mean performance across all samples
        return mean(perfs)

    def jl_eval(self):
        if self.jl.eval("length(unique(art.labels))") == 1:
            self.jl.y_hats = self.jl.eval("unique(art.labels)").repeat(len(self.jl.labels))
        else:
            self.jl.eval("y_hats = classify(art, features, get_bmu=true)")

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
            # Extract the elements of the minibatch
            data, labels, tasks = mb

            # Load the features and labels into the Julia workspace
            self.jl.features = data.squeeze().numpy().transpose()
            self.jl.labels = labels.numpy() + 1

            # Eval with timing
            start_time = time.time()
            self.jl_eval()
            end_time = time.time()
            jl_eval = end_time - start_time
            pbar.set_postfix({"jl time": jl_eval})

            # Extract the target estimates
            y_hats = self.jl.y_hats - 1
            # Append the accuracy of the minibatch
            perfs.append(accuracy_score(labels, y_hats))

        # Return the average performance across all minibatches
        return mean(perfs)
