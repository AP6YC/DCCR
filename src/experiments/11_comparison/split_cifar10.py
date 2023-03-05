# THIS MUST BE IMPORTED FIRST, AND I DON'T KNOW WHY
from src.ddvfa_foundry import DDVFAStrategy
# THIS IS IMPORTED NEXT, AND I STILL DON'T KNOW WHY
from src.datasets import SplitCIFAR10Preprocessed
# Import all utilities and scenarios
from src.utils import projectdir, print_allocated_memory, set_seed, create_default_args
from src.scenarios import fast_condensed_scenario

import torch

def ddvfa_splitcifar10(override_args=None):
    # Set the args using the continual-learning-baselines util
    args = create_default_args(
        {
            'cuda': 0,
            'epochs': 30,
            # 'learning_rate': 1e-3,
            # 'train_mb_size': 200,
            'seed': None,
            'dataset_root': None
        },
        override_args
    )

    # Set the seed everywhere for reproducibility
    set_seed(args.seed)

    # Set the device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # Create the benchmark dataset
    benchmark = SplitCIFAR10Preprocessed(
        n_experiences=5,
        # n_experiences=10,
        shuffle=False,
        # replace_existing=True,
    )

    # Create the DDVFA strategy instance
    cl_strategy = DDVFAStrategy(
        projectdir(),
        # runtime="/home/sap625/julia",
    )

    # Training Loop
    print('Starting experiment...')

    # Run the condensed scenario and get the final training and testing results
    train_results, test_results = fast_condensed_scenario(benchmark, cl_strategy)

    print("--- INDIVIDUAL RESULTS ---")
    print(test_results)
    avg_perf = sum(test_results)/len(test_results)
    print("--- FINAL TEST RESULT ---")
    print(avg_perf)
    print("--- END OF SCENARIO ---")

    return train_results, test_results, avg_perf

if __name__ == "__main__":
    train_results, test_results, avg_perf = ddvfa_splitmnist()
    print(train_results, test_results, avg_perf)
