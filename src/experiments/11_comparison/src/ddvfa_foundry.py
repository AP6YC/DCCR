import julia

runtime="/home/sap625/julia"
julia.install(julia=runtime)

jl_run = julia.Julia(
    runtime=runtime,
    compiled_modules=False,
)

from julia import Main

import time


from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import ipdb
import torch
from tqdm import tqdm

from torchvision.models import resnet50, resnet18
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.metrics import accuracy_score
from statistics import mean

def print_allocated_memory():
   print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))

class DDVFAStrategy():
    """DDVFA Strategy"""

    # def __init__(self, preprocessed=False):
    def __init__(self,
        projectdir,
        # runtime="julia",
    ):
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
            data, labels, tasks = mb
            # jl.features = self.ext_features(data)
            self.jl.features = data.squeeze().numpy().transpose()
            self.jl.labels = labels.numpy() + 1
            # ipdb.set_trace()

            start_time = time.time()
            self.jl_train()
            end_time = time.time()
            jl_eval = end_time - start_time
            pbar.set_postfix({"jl time": jl_eval})

            y_hats = self.jl.y_hats - 1
            perfs.append(accuracy_score(labels, y_hats))

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
            data, labels, tasks = mb
            # jl.features = self.ext_features(data)

            self.jl.features = data.squeeze().numpy().transpose()
            self.jl.labels = labels.numpy() + 1
            start_time = time.time()
            # jl.eval("y_hats = classify(art, features, get_bmu=true)")
            self.jl_eval()
            end_time = time.time()
            jl_eval = end_time - start_time
            pbar.set_postfix({"jl time": jl_eval})

            y_hats = self.jl.y_hats - 1
            perfs.append(accuracy_score(labels, y_hats))

        return mean(perfs)
        # j.samples = mb
        # y_hats = j.eval("classify(art, samples)")
        # accuracy_score(y_test, y_hat)

# print_allocated_memory()




        # self.preprocessed = preprocessed
        # if not self.preprocessed:
            # rn = resnet50()
            # self.weights = ResNet18_Weights.DEFAULT
            # # rn = resnet18(pretrained=True)
            # rn = resnet18(weights=self.weights)
            # self.mod = create_feature_extractor(rn, {'layer4': 'layer4'})
            # self.mod = self.mod.to('cuda')
            # self.mod.eval()
            # # self.weights = ResNet50_Weights.DEFAULT
            # self.preprocess = self.weights.transforms()
            # self.min = 0.0
            # self.max = 32.0
            # self.mult_factor = 1 / (self.max - self.min) * 2

    # def ext_features(self, img):
    #     with torch.no_grad():
    #         img = img.to('cuda')
    #         prep = self.preprocess(img)
    #         features = self.mod(prep)['layer4']
    #         # avg_features = features.mean(dim=1).flatten(start_dim=1).detach().cpu().numpy()
    #         avg_features = features.detach().mean(dim=1).flatten(start_dim=1)
    #         # avg_features = (avg_features - self.min) / (self.max - self.min) * 2 - 1
    #         avg_features = ((avg_features - self.min) * self.mult_factor - 1) * 3
    #         avg_features = avg_features.sigmoid().cpu().numpy().transpose()
    #         # avg_features = features.mean(dim=1).flatten(start_dim=1).cpu().numpy()
    #         # ipdb.set_trace()
    #         # avg_features.flatten().detach().numpy()

    #     return avg_features