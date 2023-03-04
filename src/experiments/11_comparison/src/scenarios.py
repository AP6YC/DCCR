
def iterative_scenario(benchmark: NCScenario, cl_strategy):
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

    return train_results, test_results

def condensed_scenario(benchmark: NCScenario, cl_strategy):
    test_results = []
    train_results = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)

        train_results.append(cl_strategy.train(experience))
        print('Training completed')
        print('Training results:')
        print(train_results)

    for exp_id, experience in enumerate(benchmark.test_stream):
        print("Computing accuracies")
        test_results.append(cl_strategy.eval(experience))

    return train_results, test_results
