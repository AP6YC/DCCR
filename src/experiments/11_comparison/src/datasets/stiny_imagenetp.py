from .utils import process_dataset

# import pdb

from pathlib import Path
from typing import Union, Any, Optional

from torchvision import transforms

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import TinyImagenet
from avalanche.benchmarks.generators import nc_benchmark

from torchvision.transforms import Lambda

_default_train_transform = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
        Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    ]
)

_default_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
        Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    ]
)

def _get_tiny_imagenet_dataset(dataset_root):
    train_set = TinyImagenet(root=dataset_root, train=True)

    test_set = TinyImagenet(root=dataset_root, train=False)

    return train_set, test_set

def SplitTinyImageNetPreprocessed(
    n_experiences=10,
    *,
    return_task_id=False,
    seed=0,
    fixed_class_order=None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Union[str, Path] = None,
    replace_existing: bool = False,
):
    """
    Creates a CL benchmark using the Tiny ImageNet dataset.

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

    :param n_experiences: The number of experiences in the current benchmark.
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
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'tinyimagenet' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    # Load the tiny imagenet dataset
    train_set, test_set = _get_tiny_imagenet_dataset(dataset_root)

    # pdb.set_trace()
    # Manually set the initial transforms
    train_set.transform = train_transform
    test_set.transform = eval_transform

    # Preprocess the dataset
    dataset_train, dataset_test = process_dataset(
        train_set,
        test_set,
        "tiny_imagenet",
        replace_existing,
    )

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
