import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import Linear, Sigmoid, Softmax, Dropout, Relu, Tanh, RegularizedLinear, \
    CheapTanh
from loss import SquaredLoss, CrossEntropyLoss, ClaudioMaxNLL, NLL
from optim import RMSProp, AdaGrad, MomentumSGD, SGD
from network import Seq
from trainers import PatienceTrainer, MinibatchTrainer, SimpleTrainer
from utils import chunks, partition, to_one_hot_vector_targets


def gen_data(n=300, dataset='clusters'):
    classes_n = 4
    if dataset == 'clusters':
        data, targets = datasets.make_classification(n_samples=n, n_features=2, n_informative=2, n_redundant=0,
                                                     n_classes=4, class_sep=2.5, n_clusters_per_class=1)
    elif dataset == 'circles':
        data, targets = datasets.make_circles(
            n_samples=n, shuffle=True, noise=0.1, random_state=None, factor=0.1)
    elif dataset == 'moons':
        data, targets = datasets.make_moons(n_samples=n, shuffle=True, noise=0.2)

    train_data, valid_data, test_data = partition(data, 3)
    train_targets, valid_targets, test_targets = partition(targets, 3)

    train_set = to_one_hot_vector_targets(classes_n, zip(train_data, train_targets))
    valid_set = to_one_hot_vector_targets(classes_n, zip(valid_data, valid_targets))
    test_set = to_one_hot_vector_targets(classes_n, zip(test_data, test_targets))

    return train_set, valid_set, test_set


def scatter_train_data(train_set):
    plt.figure(1)
    markers = ['o', 'x', 'v', 's']
    for class_int, marker in enumerate(markers):
        x = [point[0] for point, t in train_set if np.argmax(t) == class_int]
        y = [point[1] for point, t in train_set if np.argmax(t) == class_int]
        plt.scatter(x, y, color='black', marker=marker, s=100)


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


def plot_mean_loss(x):
    plt.figure()
    plt.title('Mean Loss Across Epoch (mean J)')
    plt.plot(xrange(len(x)), x, color='red')
    plt.ylim([0, 2])
    return plt


class PerceptronExperiment:
    def run(self):
        train_set, valid_set, test_set = gen_data(dataset='circles')

        l1 = 0.001
        l2 = 0.001
        model = Seq([
            Linear(2, 10),
            Tanh(),
            Linear(10, 4),
        ])

        # trainer = SimpleTrainer()
        # trainer.train(model, train_set,
        #               loss=CrossEntropyLoss(),
        #               optimizer=SGD(learning_rate=0.1),
        #               # optimizer=MomentumSGD(learning_rate=0.1, momentum=0.8),
        #               # optimizer=AdaGrad(learning_rate=0.9),
        #               # optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
        #               epochs=100)

        trainer = MinibatchTrainer()
        mean_losses = trainer.train_minibatches(model, train_set,
                                                batch_size=1,
                                                epochs=200,
                                                loss=CrossEntropyLoss(),
                                                optimizer=MomentumSGD(learning_rate=0.1))

        # trainer = PatienceTrainer()
        # mean_losses = trainer.train(model, train_set, valid_set, test_set,
        #                             batch_size=1,
        #                             max_epochs=10000,
        #                             loss=CrossEntropyLoss(),
        #                             test_score_function=CrossEntropyLoss.test_score,
        #                             optimizer=SGD(learning_rate=0.1))

        draw_decision_surface(model)
        scatter_train_data(train_set)

        plot_mean_loss(mean_losses)
        plt.show()


PerceptronExperiment().run()
