import gzip
import cPickle as pickle
import unittest

import numpy as np
import matplotlib.pyplot as plt

from main import SequentialModel, LinearLayer, SoftmaxLayer, NLL, SquaredLoss, PrintLayer


class Mnist(unittest.TestCase):

    def load_mnist_dataset(self):
        with gzip.open('./mnist.pkl.gz', 'rb') as f:
            try:
                return pickle.load(f, encoding='latin1')
            except:
                return pickle.load(f)

    def test_Mnist(self):

        train_set, valid_set, test_set = self.load_mnist_dataset()

        # train_set = [train_set[0][0:100], train_set[1][0:100]]

        model = SequentialModel(784, [
            LinearLayer(784, 10),
            SoftmaxLayer(),
        ])

        model.learn(
            input_data=train_set[0],
            target_data=train_set[1],
            loss=NLL(),
            epochs=10,
            learning_rate=0.1,
            show_progress=True
        )

        model.plot_errors_history()
        model.plot_loss_gradient_history()
        plt.show()