from __future__ import absolute_import

import random
import unittest

import numpy

from layers import Linear, Sigmoid
from loss import CrossEntropyLoss
from network import Seq
from optim import SGD
from trainers import SimpleTrainer
from utils import to_one_hot_vector_targets

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class IrisTest(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.train_set = to_one_hot_vector_targets(classes_n=3, train_set=zip(iris.data, iris.target))
        random.shuffle(self.train_set, random=lambda: 0.1)

    def plot_loss_history(self, x):
        plt.figure()
        plt.title('Epoch Mean Loss history')
        plt.plot(xrange(len(x)), x, color='red')
        plt.ylim([0, 1])

    def test_iris(self):
        scores = []
        for i in range(1):
            hidden = 50
            l1 = Linear(4, hidden, initialize='ones')
            l2 = Linear(hidden, 3, initialize='ones')
            l1.W *= 0.000000001
            l2.W *= 0.00000001
            model = Seq(l1, Sigmoid, l2)
            loss = CrossEntropyLoss()

            trainer = SimpleTrainer()
            losses = trainer.train(model, self.train_set,
                                   epochs=100,
                                   loss=loss,
                                   optimizer=SGD(learning_rate=0.01))

            score = loss.test_score(model, self.train_set)
            print("hidden=%f score=%f" % (hidden, score))
            scores.append(score)
            self.plot_loss_history(losses)
            plt.show()

        self.assertGreaterEqual(numpy.mean(scores), 94.)
