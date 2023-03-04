# Import all of the names from the file that we want to extend
from avalanche.benchmarks.classic.cmnist import *
from .feature_extractor import FeatureExtractor
from torchvision.transforms import Lambda
import ipdb
from torch.utils.data import TensorDataset


from pathlib import Path
from typing import Optional, Sequence, Union, Any
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import (
    ToTensor,
    ToPILImage,
    Compose,
    Normalize,
    RandomRotation,
)
import numpy as np

from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets.external_datasets.mnist import \
    get_mnist_dataset




_default_mnist_train_transform = Compose(
    [Normalize((0.1307,), (0.3081,))]
)

_default_mnist_eval_transform = Compose(
    [Normalize((0.1307,), (0.3081,))]
)


def SplitMNISTPreprocessed(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform: Optional[Any] = _default_mnist_train_transform,
    eval_transform: Optional[Any] = _default_mnist_eval_transform,
    dataset_root: Union[str, Path] = None,
    replace_existing: bool = False,
):
    """
    Creates a CL benchmark using the MNIST dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    # Create the custom feature extractor
    fe = FeatureExtractor()

    # Point to the directory containing the model files and create it if necessary
    file_dir = Path("models")
    file_dir.mkdir(parents=True, exist_ok=True)

    # Declare the filenames
    train_file = file_dir.joinpath("mnist_train.pt")
    train_targets_file = file_dir.joinpath("mnist_train_targets.pt")
    test_file = file_dir.joinpath("mnist_test.pt")
    test_targets_file = file_dir.joinpath("mnist_test_targets.pt")

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
        # Load the dataset
        mnist_train, mnist_test = get_mnist_dataset(dataset_root)

        # Set the transform to grayscale-to-RGB for the feature extractor
        trans = Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        mnist_train.transform = trans
        mnist_test.transform = trans

        # Get the features (and statistics) from the training dataset
        features_train = fe.process(mnist_train)
        # Use the training statistics for transforming the test dataset
        features_test = fe.process(mnist_test)
        # Save
        torch.save(features_train, train_file)
        torch.save(features_test, test_file)

        targets_train = mnist_train.targets
        targets_test = mnist_test.targets
        torch.save(targets_train, train_targets_file)
        torch.save(targets_test, test_targets_file)

    # ipdb.set_trace()
    dataset_train = TensorDataset(features_train, targets_train)
    dataset_test = TensorDataset(features_test, targets_test)

    return nc_benchmark(
        train_dataset=dataset_train,
        test_dataset=dataset_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        train_transform=None,
        eval_transform=None,
        # train_transform=train_transform,
        # eval_transform=eval_transform,
    )
