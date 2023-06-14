# Include the above folder
import include_src

from avalanche.benchmarks import SplitTinyImageNet
from src.utils import scratchdir

# Get the scratch directory
dataset_root = scratchdir("tiny_imagenet")

benchmark = SplitTinyImageNet(
    n_experiences=10,
    return_task_id=True,
    dataset_root=dataset_root,
)
