import gzip
import cPickle as pickle
import unittest

import numpy as np
import matplotlib.pyplot as plt

from layers import LinearLayer, SoftmaxLayer, ReluLayer
from loss import SoftmaxNLL
from optim import RMSProp, MomentumSGD, AdaGrad
from sequential import SequentialModel


class Mnist(unittest.TestCase):
    def load_mnist_dataset(self):
        with gzip.open('./mnist.pkl.gz', 'rb') as f:
            try:
                return pickle.load(f, encoding='latin1')
            except:
                return pickle.load(f)

    def test_Mnist(self):
        train_set, valid_set, test_set = self.load_mnist_dataset()

        # train_set = [train_set[0][0:1000], train_set[1][0:1000]]

        model = SequentialModel([
            LinearLayer(784, 10),
        ])

        # model.learn(
        #     input_data=train_set[0],
        #     target_data=train_set[1],
        #     loss=SoftmaxNLL(),
        #     epochs=3,
        #
        #     optimizer=RMSProp(learning_rate=0.1, decay_rate=0.9),
        #
        #     show_progress=True,
        #     save_progress=True
        # )

        model.learn_minibatch(
            input_data=train_set[0],
            target_data=train_set[1],
            batch_size=600,
            loss=SoftmaxNLL(),
            epochs=1000,
            # optimizer=RMSProp(learning_rate=0.13, decay_rate=0.9),
            optimizer=AdaGrad(learning_rate=0.9),
            show_progress=True,
            save_progress=True
        )

        model.test(input_data=test_set[0], target_data=test_set[1], loss=SoftmaxNLL())

        model.save_to_file('mnist-b600-e1000-adagrad.pkl')

        # model.plot_errors_history()
        # model.plot_loss_gradient_history()
        # plt.show()
