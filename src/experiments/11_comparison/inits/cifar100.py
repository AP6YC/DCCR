import include_src

from avalanche.benchmarks.classic import SplitCIFAR100
from src.utils import scratchdir

# Get the scratch directory
dataset_root = scratchdir("cifar100")

# Load the dataset
benchmark = SplitCIFAR100(
    n_experiences=10,
    return_task_id=True,
)
