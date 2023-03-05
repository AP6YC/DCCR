# THIS MUST BE IMPORTED FIRST, AND I DON'T KNOW WHY
from src.ddvfa_foundry import DDVFAStrategy
# THIS IS IMPORTED NEXT, AND I STILL DON'T KNOW WHY
from src.datasets.smnistp import SplitMNISTPreprocessed
# Import all utilities
from utils import projectdir, print_allocated_memory, set_seed

benchmark = SplitMNISTPreprocessed(
    # n_experiences=5,
    n_experiences=10,
    shuffle=False,
    # replace_existing=True,
)

# Create the Strategy Instance
cl_strategy = DDVFAStrategy(
    projectdir(),
    runtime="/home/sap625/julia",
)

# Training Loop
print('Starting experiment...')

# Run the condensed scenario and get the final training and testing results
train_results, test_results = fast_condensed_scenario(benchmark, cl_strategy)

# test_results = []
# train_results = []
# for exp_id, experience in enumerate(benchmark.train_stream):
#     print("Start of experience ", experience.current_experience)

#     train_results.append(cl_strategy.train(experience))
#     print('Training completed')
#     print('Training results:')
#     print(train_results)

# for exp_id, experience in enumerate(benchmark.test_stream):
#     print("Computing accuracies")
#     test_results.append(cl_strategy.eval(experience))

print("--- INDIVIDUAL RESULTS ---")
print(test_results)
avg_perf = sum(test_results)/len(test_results)
print("--- FINAL TEST RESULT ---")
print(avg_perf)

# print(test_results, train_results)
