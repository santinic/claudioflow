import itertools
import timeit

import matplotlib.pyplot as plt


def analyse(train_function, **kwargs):
    # Make batch_size=10 behave the same as batch_size=[10]
    kwargs_only_lists = []
    for k, v in kwargs.items():
        v = v if type(v) == list else [v]
        kwargs_only_lists.append((k, v))

    # A list of all possible variable name/values
    param_values = [[(k, i) for i in v] for k, v in kwargs_only_lists]

    # Go through all the possible combinations of variable values
    results = []
    for combination in itertools.product(*param_values):
        combination_dict = dict(combination)
        print('%s training...' % combination_dict)

        start_time = timeit.default_timer()
        outputs = train_function(**combination_dict)
        end_time = timeit.default_timer()
        outputs['time'] = (end_time - start_time) / 60.

        result = (combination_dict, outputs)
        results.append(result)
    return results


def plot_analyser_results(results):
    for r in results:
        print(r)

# def f(learning_rate, batch_size):
#     return { 'train_score': 100, 'test_score': 200 }
