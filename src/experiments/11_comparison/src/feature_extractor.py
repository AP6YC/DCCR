# mnist_train.data.shape
import torch
# import ipdb
from torchvision.transforms import Normalize
from torchvision.models import resnet50, resnet18
from torchvision.models import ResNet50_Weights, ResNet18_Weights
# from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm

# import ipdb
# import pdb

class FeatureExtractor():

    def __init__(self,
        layer: str = 'layer4',
        sigmoid_scaling: float = 3.0,
    ):
        self.sigmoid_scaling = sigmoid_scaling

        # Create the extraction network
        # self.weights = ResNet50_Weights.DEFAULT
        # rn = resnet50()
        self.weights = ResNet18_Weights.DEFAULT
        rn = resnet18(weights=self.weights)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.layer = 'layer4'
        self.layer = layer

        # Use torch FX and load to the device
        self.mod = create_feature_extractor(rn, {self.layer: self.layer})
        self.mod = self.mod.to(self.device)
        # Evaluation model
        self.mod.eval()
        # Preprocess stage
        self.preprocess = self.weights.transforms()

        # Z-score parameters
        self.mean = None
        self.std = None

    @torch.no_grad()
    def ext_features(self, img):
        img = img.to(self.device)
        prep = self.preprocess(img)
        features = self.mod(prep)[self.layer]
        return features

    @torch.no_grad()
    def avg_features(self, features):
        features = features.detach().mean(dim=1).flatten(start_dim=1)
        return features

    @torch.no_grad()
    def process(self, dataset):
        data_loader = DataLoader(
            dataset=dataset,
            pin_memory=True,
            batch_size=256,
            shuffle=False,
        )

        outs = []
        for mb in tqdm(data_loader):
            data, labels = mb
            # pdb.set_trace()
            features = self.ext_features(data)
            # ipdb.set_trace()
            features = self.avg_features(features)
            outs.append(features)

        # Create a new tensor for the averaged features of all samples
        outs = torch.cat(outs)
        # ipdb.set_trace()

        # Set the stats from the averaged features of the dataset
        if (self.mean is None) and (self.std is None):
            self.set_mean_std(outs)

        # outs = self.norm_features(outs)
        outs = self.z_score(outs)
        outs = self.sigmoid_squash(outs)

        # Make sure the data goes back to the cpu regardless
        outs = outs.cpu()

        return outs

    @torch.no_grad()
    def z_score(self, features):
        return (features - self.mean) / self.std

    @torch.no_grad()
    def set_mean_std(self, features):
        self.mean, self.std = features.mean((0,)), features.std((0,))

    @torch.no_grad()
    def sigmoid_squash(self, features):
        features = features * self.sigmoid_scaling
        return features.sigmoid()

# fe = FeatureExtractor()
