from src.utils import scratchdir

# Get the scratch directory
dataset_root = scratchdir("tiny_imagenet")

from avalanche.benchmarks.classic import SplitCIFAR10

# CL Benchmark Creation
benchmark = SplitCIFAR10(
    n_experiences=5,
    return_task_id=True,
    dataset_root=dataset_root,
)
