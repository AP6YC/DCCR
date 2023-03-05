from src.utils import scratchdir

# Get the scratch directory
dataset_root = scratchdir("tiny_imagenet")

benchmark = avl.benchmarks.SplitTinyImageNet(
    n_experiences=10,
    return_task_id=True,
    dataset_root=dataset_root
)
