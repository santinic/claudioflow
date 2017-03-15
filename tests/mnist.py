import gzip
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import analyser
from layers import LinearLayer, SoftmaxLayer, ReluLayer, RegularizedLinearLayer
from loss import CrossEntropyLoss
from optim import RMSProp, MomentumSGD, AdaGrad
from sequential import SequentialModel
from trainers import MinibatchTrainer, PatienceTrainer
from utils import slice_percentage


class MnistTrainer:
    def __init__(self):
        train_set, valid_set, test_set = self.load_mnist_dataset()
        # train_set = [train_set[0][0:1000], train_set[1][0:1000]]
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

    def run(self, batch_size=10, learning_rate=0.8, train_set_percentage=1.0, epochs=3):
        model = SequentialModel([
            LinearLayer(784, 10, initialize='random'),
        ])

        train_set_sliced = slice_percentage(self.train_set, train_set_percentage)

        trainer = MinibatchTrainer()
        trainer.train_minibatches(model,
                                  train_set_sliced,
                                  batch_size=batch_size,
                                  loss=CrossEntropyLoss(),
                                  epochs=epochs,
                                  # optimizer=MomentumSGD(learning_rate=0.8, momentum=0.8),
                                  # optimizer=RMSProp(learning_rate=learning_rate, decay_rate=0.9),
                                  optimizer=AdaGrad(learning_rate=learning_rate),
                                  show_progress=True)

        train_score, test_score = self.get_score(model, self.train_set, self.test_set)

        return {
            'train_score': train_score,
            'test_score': test_score,
        }

        # trainer = PatienceTrainer()
        # trainer.train(model,
        #               train_set, valid_set, test_set,
        #               batch_size=20,
        #               loss=CrossEntropyLoss(),
        #               max_epochs=1000,
        #               # optimizer=RMSProp(learning_rate=0.13, decay_rate=0.9),
        #               optimizer=AdaGrad(learning_rate=0.9)
        # )

        # self.print_score(model, train_set[0], train_set[1], test_set[0], test_set[1])

        # self.show_mnist_grid(model, test_set)
        # model.save_to_file('mnist-b600-e1000-adagrad.pkl')
        # plt.show()

    def load_mnist_dataset(self):
        with gzip.open('./mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
            train_set = zip(train_set[0], train_set[1])
            valid_set = zip(valid_set[0], valid_set[1])
            test_set = zip(test_set[0], test_set[1])
            return train_set, valid_set, test_set

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

    def get_score(self, model, train_set, test_set):
        train_err = 0
        for x, target in train_set:
            if np.argmax(model.forward(x)) != target:
                train_err += 1
        train_score = (1.0 - train_err / float(len(train_set))) * 100.0

        test_err = 0
        for x, target in test_set:
            if np.argmax(model.forward(x)) != target:
                test_err += 1
        test_score = (1.0 - test_err / float(len(test_set))) * 100.0
        return train_score, test_score


# MnistTrainer().run()

results = analyser.analyse(MnistTrainer().run,
                           learning_rate=[1, 0.8, 0.5], batch_size=10, epochs=[2,4], train_set_percentage=1)
analyser.plot_analyser_results(results)
