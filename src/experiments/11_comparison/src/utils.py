from pathlib import Path
import torch
import random
import numpy as np

# # Point to the top of the project relative to this script
def projectdir(*args):
    return str(Path.cwd().joinpath("..", "..", "..", *args).resolve())

def print_allocated_memory():
   print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))

def scratchdir():
    dataset_root = Path("/lustre", "scratch", "sap625", "core50")
    dataset_root.mkdir(parents=True, exist_ok=True)
    return dataset_root

def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False