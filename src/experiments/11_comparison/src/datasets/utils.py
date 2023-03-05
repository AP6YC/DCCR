from ..feature_extractor import FeatureExtractor

# from .feature_extractor import FeatureExtractor
# import ipdb
# import pdb

from torchvision.transforms import Lambda
from torch.utils.data import TensorDataset

from pathlib import Path
import torch

def process_dataset(raw_train, raw_test, name, replace_existing):
    # Create the custom feature extractor
    fe = FeatureExtractor()

    # Point to the directory containing the model files and create it if necessary
    model_dir = Path("models")
    file_dir = model_dir.joinpath(name)
    file_dir.mkdir(parents=True, exist_ok=True)

    # Declare the filenames
    train_file = file_dir.joinpath(f"{name}_train.pt")
    train_targets_file = file_dir.joinpath(f"{name}_train_targets.pt")
    test_file = file_dir.joinpath(f"{name}_test.pt")
    test_targets_file = file_dir.joinpath(f"{name}_test_targets.pt")

    # If we have the files, simply load
    if train_file.is_file() and test_file.is_file() and not replace_existing:
        # Load the training data and targets
        features_train = torch.load(train_file)
        targets_train = torch.load(train_targets_file)

        # Load the test targets and targets
        features_test = torch.load(test_file)
        targets_test = torch.load(test_targets_file)
    # Otherwise, generate the features and save
    else:
        # Set the transform to grayscale-to-RGB for the feature extractor
        # trans = Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        # raw_train.transform = trans
        # raw_test.transform = trans

        # Get the features (and statistics) from the training dataset
        features_train = fe.process(raw_train)
        # Use the training statistics for transforming the test dataset
        features_test = fe.process(raw_test)
        # Save
        torch.save(features_train, train_file)
        torch.save(features_test, test_file)

        # targets_train = raw_train.targets
        # targets_test = raw_test.targets
        # Cast target to Tensors in case they are loaded as lists
        targets_train = torch.IntTensor(raw_train.targets)
        targets_test = torch.IntTensor(raw_test.targets)
        torch.save(targets_train, train_targets_file)
        torch.save(targets_test, test_targets_file)

    # ipdb.set_trace()
    dataset_train = TensorDataset(features_train, targets_train)
    dataset_test = TensorDataset(features_test, targets_test)

    return dataset_train, dataset_test
