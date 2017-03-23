import gzip
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import analyser
from layers import Linear, Softmax, Relu, RegularizedLinear
from loss import CrossEntropyLoss
from optim import RMSProp, MomentumSGD, AdaGrad
from network import Seq
from trainers import MinibatchTrainer, PatienceTrainer
from utils import slice_percentage


class MnistExperiment:
    def __init__(self):
        train_set, valid_set, test_set = self.load_mnist_dataset()
        # train_set = [train_set[0][0:1000], train_set[1][0:1000]]

        # make all the targets, one-hot-vectors. For example 3 becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        classes_n = 10
        # print(train_set[0][1])
        self.make_one_hot_vectors(classes_n, train_set)
        self.make_one_hot_vectors(classes_n, valid_set)
        self.make_one_hot_vectors(classes_n, test_set)
        # print(train_set[0][1])

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

    @staticmethod
    def make_one_hot_vectors(classes_n, train_set):
        for i, (x, target_class) in enumerate(train_set):
            train_set[i] = (x, CrossEntropyLoss.make_one_hot_target(classes_n, target_class))

    def run(self, batch_size=10, learning_rate=0.6, train_set_percentage=1.0, epochs=3):
        model = Seq([
            Linear(784, 10, initialize='random'),
        ])

        train_set_sliced = slice_percentage(self.train_set, train_set_percentage)

        trainer = MinibatchTrainer()
        trainer.train_minibatches(model,
                                  train_set_sliced,
                                  batch_size=batch_size,
                                  loss=CrossEntropyLoss(),
                                  epochs=epochs,
                                  optimizer=MomentumSGD(learning_rate=learning_rate, momentum=0.5),
                                  # optimizer=RMSProp(learning_rate=learning_rate, decay_rate=0.9),
                                  # optimizer=AdaGrad(learning_rate=learning_rate),
                                  show_progress=True)

        self.show_mnist_grid(model, self.test_set)

        # trainer = PatienceTrainer()
        # trainer.train(model,
        #               train_set_sliced, self.valid_set, self.test_set,
        #               batch_size=batch_size,
        #               loss=CrossEntropyLoss(),
        #               max_epochs=100,
        #               # optimizer=MomentumSGD(learning_rate=learning_rate, momentum=0.5),
        #               optimizer=RMSProp(learning_rate=learning_rate, decay_rate=0.9),
        #               # optimizer=AdaGrad(learning_rate=learning_rate),
        #               test_score_function=self.test_score_fun
        # )

        test_score = CrossEntropyLoss().test_score(model, self.test_set)

        return {
            # 'train_score': train_score,
            'test_score': test_score,
        }

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


experiment = MnistExperiment()
results = analyser.analyse(experiment.run,
                           learning_rate=[0.1], batch_size=60, epochs=[20], train_set_percentage=1)
analyser.plot_analyser_results(results)
