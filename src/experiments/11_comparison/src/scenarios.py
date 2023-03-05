
def iterative_scenario(benchmark: NCScenario, cl_strategy) -> tuple[list, list]:
    """Iteratively trains and tests on one experience at a time.

    Args:
        benchmark (NCScenario): the avalanche benchmark containing a `train_stream` and `test_stream`.
        cl_strategy (_type_): the avalanche-style continual learning strategy.

    Returns:
        tuple[list, list]: the train and test results as lists of floats corresponding to experience-wise accuracies.
    """

    test_results = []
    train_results = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)

        train_results.append(cl_strategy.train(experience))
        print("Training completed")
        print("Training results:")
        print(train_results)

        print("Computing accuracy on the current test set")
        test_results.append(cl_strategy.eval(benchmark.test_stream[exp_id]))
        print("Testing completed")
        print("Testing results:")
        print(test_results)

    return train_results, test_results

def fast_condensed_scenario(benchmark: NCScenario, cl_strategy) -> tuple[list, list]:
    """This is a condensed scenario that only tests the performance on all classes after training.

    Args:
        benchmark (NCScenario): the avalanche benchmark containing a `train_stream` and `test_stream`.
        cl_strategy (_type_): the avalanche-style continual learning strategy.

    Returns:
        tuple[list, list]: the train and test results as lists of floats corresponding to experience-wise accuracies.
    """

    train_results = []
    test_results = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)

        train_results.append(cl_strategy.train(experience))
        print("Training completed")
        print("Training results:")
        print(train_results)

    for exp_id, experience in enumerate(benchmark.test_stream):
        print("Computing accuracies")
        test_results.append(cl_strategy.eval(experience))
        print("Testing results:")
        print(test_results)

    return train_results, test_results
