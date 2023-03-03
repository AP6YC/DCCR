import julia
julia.install()

from pathlib import Path

# # Point to the top of the project relative to this script
def projectdir(*args):
    return str(Path.cwd().joinpath("..", "..", "..", *args).resolve())

def print_allocated_memory():
   print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))




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

    def train(self, experience):
        train_dataset = experience.dataset
        t = experience.task_label
        train_data_loader = DataLoader(
            dataset=train_dataset,
            pin_memory=True,
            # batch_size=90,
            batch_size=256,
        )
        print(experience.dataset.__len__())
        for mb in tqdm(train_data_loader):
            data, labels, tasks = mb
            # jl.features = self.ext_features(data)
            jl.features = data.numpy().transpose()
            jl.labels = labels.numpy()
            # ipdb.set_trace()
            jl.eval("train!(art, features, y=labels)")

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
        perfs = []
        for mb in tqdm(eval_data_loader):
            data, labels, tasks = mb
            # jl.features = self.ext_features(data)
            jl.features = data.numpy().transpose()
            jl.eval("y_hats = classify(art, features, get_bmu=true)")
            y_hats = jl.y_hats
            # ipdb.set_trace()
            perfs.append(accuracy_score(labels, y_hats))
            # correct += torch.sum(y_hats == labels)
            # total += len(data)

        return mean(perfs)
        # j.samples = mb
        # y_hats = j.eval("classify(art, samples)")
        # accuracy_score(y_test, y_hat)

# print_allocated_memory()