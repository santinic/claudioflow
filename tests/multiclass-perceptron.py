import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SigmoidLayer, SoftmaxLayer, DropoutLayer, ReluLayer, TanhLayer, RegularizedLinearLayer
from loss import SquaredLoss, CrossEntropyLoss, ClaudioMaxNLL, NLL
from optim import RMSProp, AdaGrad, MomentumSGD, SGD
from sequential import SequentialModel
from trainers import PatienceTrainer, MinibatchTrainer, SimpleTrainer
from utils import chunks, partition


def gen_data():
    n = 200

    # N clusters:
    # data, targets = datasets.make_classification(
    #     n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_classes=4, class_sep=2.5, n_clusters_per_class=1)

    # Circles:
    data, targets = datasets.make_circles(
        n_samples=n, shuffle=True, noise=0.1, random_state=None, factor=0.1)

    # Moons:
    # data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.2)

    train_data, valid_data, test_data = partition(data, 3)
    train_targets, valid_targets, test_targets = partition(targets, 3)

    return train_data, train_targets, valid_data, valid_targets, test_data, test_targets


def scatter_train_data(train_data, train_targets):
    x0 = [point[0] for i, point in enumerate(train_data) if train_targets[i] == 0]
    y0 = [point[1] for i, point in enumerate(train_data) if train_targets[i] == 0]
    x1 = [point[0] for i, point in enumerate(train_data) if train_targets[i] == 1]
    y1 = [point[1] for i, point in enumerate(train_data) if train_targets[i] == 1]
    x2 = [point[0] for i, point in enumerate(train_data) if train_targets[i] == 2]
    y2 = [point[1] for i, point in enumerate(train_data) if train_targets[i] == 2]
    x3 = [point[0] for i, point in enumerate(train_data) if train_targets[i] == 3]
    y3 = [point[1] for i, point in enumerate(train_data) if train_targets[i] == 3]

    plt.figure(1)
    plt.scatter(x0, y0, color='black', marker="o", s=100)
    plt.scatter(x1, y1, color='black', marker="x", s=100)
    plt.scatter(x2, y2, color='black', marker="v", s=100)
    plt.scatter(x3, y3, color='black', marker="s", s=100)


def scatter_test_data(data, target, model):
    plt.figure(1)
    colors = ['r', 'g', 'b', 'y', 'o']
    color = lambda t: colors[t]

    for x, target in izip(data, target):
        y = model.forward(x)
        target_class = np.argmax(y)
        plt.scatter(x[0], x[1], s=100, c=color(target_class))


def predict(model, xs):
    ys = model.forward_all(xs)
    ys = np.argmax(ys, axis=1)
    return ys


def draw_decision_surface(model):
    x_min, x_max = -5., 5.
    y_min, y_max = -5., 5.
    step = .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # cs = plt.contourf(xx, yy, Z, cmap='Paired')
    plt.set_cmap(plt.cm.Paired)
    plt.pcolormesh(xx, yy, Z)


class Perceptron(unittest.TestCase):
    def test_Perceptron(self):
        train_data, train_targets, valid_data, valid_targets, test_data, test_targets = gen_data()

        l1 = 0.001
        l2 = 0.001
        model = SequentialModel([
            # LinearLayer(2, 5, initialize='random', ),
            RegularizedLinearLayer(2, 50, initialize='random', l1=l1, l2=l2),
            TanhLayer(),
            # DropoutLayer(0.6),
            RegularizedLinearLayer(50, 4, initialize='random', l1=l1, l2=l2),
        ])

        trainer = SimpleTrainer()
        trainer.train(model,
                      train_data=train_data,
                      train_targets=train_targets,
                      loss=CrossEntropyLoss(),
                      # optimizer=SGD(learning_rate=0.1),
                      optimizer=MomentumSGD(learning_rate=0.1, momentum=0.8),
                      # optimizer=AdaGrad(learning_rate=0.9),
                      # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
                      epochs=100)

        # trainer = MinibatchTrainer()
        # trainer.train_minibatches(model,
        #                           train_data, train_targets,
        #                           batch_size=10,
        #                           epochs=100,
        #                           loss=CrossEntropyLoss(),
        #                           optimizer=MomentumSGD(learning_rate=0.1, momentum=0.9))

        # trainer = PatienceTrainer()
        # trainer.train(model,
        #               train_data, train_targets, valid_data, valid_targets, test_data, test_targets,
        #               batch_size=10,
        #               max_epochs=100,
        #               loss=CrossEntropyLoss(),
        #               optimizer=MomentumSGD(learning_rate=0.1, momentum=0.9))

        draw_decision_surface(model)
        scatter_train_data(train_data, train_targets)
        # scatter_test_data(test_data, test_targets, model)

        # model.plot_errors_history()
        # model.plot_loss_gradient_history()
        plt.show()
