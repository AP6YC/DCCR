# Include the above folder
import include_src

# from avalanche.benchmarks import SplitTinyImageNet
from src.utils import scratchdir
from src.datasets.ctiny_imagenetp import SplitTinyImageNetPreprocessed

# Get the scratch directory
dataset_root = scratchdir("tiny_imagenet")

# Load the
benchmark = SplitTinyImageNetPreprocessed(
    n_experiences=10,
    return_task_id=True,
    dataset_root=dataset_root,
    replace_existing=False
)
