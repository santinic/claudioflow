import unittest
from itertools import izip

import matplotlib.pyplot as plt
from sklearn import datasets

from layers import Linear, Sigmoid
from loss import SquaredLoss
from optim import RMSProp, AdaGrad, MomentumSGD, SGD
from network import Seq
from trainers import SimpleTrainer

OUTPUT_A = 1.
OUTPUT_B = 0.
MIDDLE = 0.5


def gen_data():
    n = 100

    # Two clusters:
    # data, targets = datasets.make_classification(
    #     n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_classes=2, class_sep=2.0, n_clusters_per_class=1)

    # Circles:
    # data, targets = datasets.make_circles(
    #     n_samples=n, shuffle=True, noise=0.05, random_state=None, factor=0.1)

    # Moons:
    data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05)

    train_data, test_data = data[:n / 2], data[n / 2:]
    train_targets, test_targets = targets[:n / 2], targets[n / 2:]

    # Change targets to arbitrary values
    train_targets = [OUTPUT_A if t == 1 else OUTPUT_B for t in train_targets]
    test_targets = [OUTPUT_A if t == 1 else OUTPUT_B for t in test_targets]

    # Scatter train_data
    x0 = [point[0] for i, point in enumerate(train_data) if train_targets[i] == 0]
    y0 = [point[1] for i, point in enumerate(train_data) if train_targets[i] == 0]
    x1 = [point[0] for i, point in enumerate(train_data) if train_targets[i] == 1]
    y1 = [point[1] for i, point in enumerate(train_data) if train_targets[i] == 1]

    plt.figure(1)
    plt.scatter(x0, y0, color='gray', marker="o", s=100)
    plt.scatter(x1, y1, color='gray', marker="x", s=100)

    return zip(train_data, train_targets), zip(test_data, test_targets)


def scatter_test_data(test_set, model):
    plt.figure(1)
    color = lambda y: 'b' if y > MIDDLE else 'r'

    for x, target in test_set:
        y = model.forward(x)
        # print(y, y > MIDDLE)
        plt.scatter(x[0], x[1], s=40, c=color(y))


class Perceptron(unittest.TestCase):
    def test_Perceptron(self):
        train_set, test_set = gen_data()

        model = Seq([
            Linear(2, 5, initialize='random'),
            Sigmoid(),
            Linear(5, 1, initialize='random'),
            Sigmoid(),
        ])

        SimpleTrainer().train(model,
                              train_set=train_set,
                              loss=SquaredLoss(),
                              # optimizer=SGD(learning_rate=0.1),
                              optimizer=MomentumSGD(learning_rate=0.1, momentum=0.9),
                              # optimizer=AdaGrad(learning_rate=0.9),
                              # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),

                              epochs=200,
                              save_progress=False)

        # model.learn_minibatch(
        # input_data=train_data,
        # target_data=train_targets,
        # loss=SquaredLoss(),
        # batch_size=5,
        # # optimizer=SGD(learning_rate=0.1),
        # # optimizer=MomentumSGD(learning_rate=0.1, momentum=0.9),
        # optimizer=AdaGrad(learning_rate=0.9),
        # # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
        #
        # epochs=100,
        # save_progress=True)

        model.save_to_file('perceptron.pkl')

        scatter_test_data(test_set, model)

        # model.plot_errors_history()
        # model.plot_loss_gradient_history()
        plt.show()
