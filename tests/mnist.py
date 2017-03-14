import gzip
import cPickle as pickle
import unittest

import numpy as np
import matplotlib.pyplot as plt

from layers import LinearLayer, SoftmaxLayer, ReluLayer
from loss import CrossEntropyLoss
from optim import RMSProp, MomentumSGD, AdaGrad
from sequential import SequentialModel
from trainers import MinibatchTrainer, PatienceTrainer


class Mnist(unittest.TestCase):
    def test_Mnist(self):
        train_set, valid_set, test_set = self.load_mnist_dataset()

        # train_set = [train_set[0][0:1000], train_set[1][0:1000]]

        model = SequentialModel([
            LinearLayer(784, 10),
        ])

        trainer = PatienceTrainer()
        trainer.train(model,
                      train_set, valid_set, test_set,
                      batch_size=20,
                      loss=CrossEntropyLoss(),
                      max_epochs=1000,
                      # optimizer=RMSProp(learning_rate=0.13, decay_rate=0.9),
                      optimizer=AdaGrad(learning_rate=0.9)
        )

        self.show_mnist_grid(model, test_set)

        # model.save_to_file('mnist-b600-e1000-adagrad.pkl')

        plt.show()

    def load_mnist_dataset(self):
        with gzip.open('./mnist.pkl.gz', 'rb') as f:
            try:
                return pickle.load(f, encoding='latin1')
            except:
                return pickle.load(f)

    def show_mnist_grid(self, model, test_set, n=10):
        test_data = test_set[0]
        fig, ax = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in xrange(n ** 2):
            y = model.forward(test_data[i])
            output = np.argmax(y)
            img = test_data[i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('%d' % output)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
