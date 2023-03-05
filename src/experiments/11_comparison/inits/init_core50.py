from avalanche.benchmarks import CORe50
from torchvision import transforms
from src.utils import scratchdir

_mu = [0.485, 0.456, 0.406]  # imagenet normalization
_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_mu,
                            std=_std)
])

# Get the scratch directory
dataset_root = scratchdir("core50")

# benchmark_instance = CORe50(scenario="nicv2_79", mini=False)
benchmark = CORe50(
    scenario='nc',
    train_transform=transform,
    eval_transform=transform,
    dataset_root=dataset_root,
)
