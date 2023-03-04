# THIS MUST BE IMPORTED FIRST, AND I DON'T KNOW WHY
from src.ddvfa_foundry import DDVFAStrategy
# THIS IS IMPORTED NEXT, AND I STILL DON'T KNOW WHY
from src.datasets.smnistp import SplitMNISTPreprocessed

from utils import projectdir, print_allocated_memory

# from pathlib import Path

# # # Point to the top of the project relative to this script
# def projectdir(*args):
#     return str(Path.cwd().joinpath("..", "..", "..", *args).resolve())

# def print_allocated_memory():
#    print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))

benchmark = SplitMNISTPreprocessed(
    # n_experiences=5,
    n_experiences=10,
    shuffle=False,
    # replace_existing=True,
)

# Create the Strategy Instance
cl_strategy = DDVFAStrategy(
    projectdir(),
    # runtime="/home/sap625/julia",
)

# Training Loop
print('Starting experiment...')

test_results = []
train_results = []
for exp_id, experience in enumerate(benchmark.train_stream):
    print("Start of experience ", experience.current_experience)

    train_results.append(cl_strategy.train(experience))
    print('Training completed')
    print('Training results:')
    print(train_results)

    print('Computing accuracy on the current test set')
    test_results.append(cl_strategy.eval(benchmark.test_stream[exp_id]))
    print('Testing completed')
    print('Testing results:')
    print(test_results)
    # results.append(cl_strategy.eval(scenario.test_stream))

print(test_results, train_results)



# print(jl.eval("AdaptiveResonance.get_n_weights(art)"))
# print(jl.eval("size(art.F2)"))
# print(jl.eval("art.labels"))
# print(1 == jl.eval("length(unique(art.labels))"))
# print(type(jl.eval("unique(art.labels)")))
