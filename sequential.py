import random

import numpy as np
import matplotlib.pyplot as plt
from future.moves.itertools import zip_longest
from itertools import izip
import cPickle as pickle


class SequentialModel:
    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers
        self.errors_history = []
        self.loss_gradient_history = []

    def add(self, layer):
        self.layers.append(layer)

    def validate_input_data(self, x):
        if type(x) == list:
            raise Exception("Input shouldn't be a Python list, but a numpy.ndarray.")

    def forward(self, x, is_training=False):
        self.validate_input_data(x)

        y = None
        for i, layer in enumerate(self.layers):
            y = layer.forward(x, is_training)
            x = y
        return y

    def backward(self, dJdy, optimizer=None, update=True):
        for layer in reversed(self.layers):
            dJdy_for_previous_layer = layer.backward(dJdy)

            # If the layer has some internal state we update it
            if hasattr(layer, 'update_gradient') and update and (optimizer is not None):
                update_gradient = layer.update_gradient(dJdy)
                optimizer.update(layer, update_gradient)

            dJdy = dJdy_for_previous_layer
        return dJdy

    def learn_one(self, x, target, loss, optimizer, show_progress=False):
        y = self.forward(x, is_training=True)
        J = loss.calc_loss(y, target)
        dJdy = loss.calc_gradient(y, target)
        self.backward(dJdy, optimizer)
        return J, dJdy

    def learn(self, input_data, target_data, loss, epochs, optimizer, show_progress=False, save_progress=False):
        '''This is "online learning": we backpropagate after forwarding one input_data at the time.'''
        for epoch in xrange(epochs):
            if show_progress: print("Epoch %d/%d" % (epoch + 1, epochs))
            input_count = 0
            for x, target in izip(input_data, target_data):
                self.show_progress(show_progress, input_count, input_data)
                J, dJdy = self.learn_one(x, target, loss, optimizer, show_progress=show_progress)
                if save_progress:
                    self.errors_history.append(J)
                    self.loss_gradient_history.append(dJdy)
                input_count += 1

    def learn_minibatch(self, input_data, target_data, batch_size, loss, epochs, optimizer, show_progress=False,
                        save_progress=False):
        def chunks(iterable, n):
            "Collect data into fixed-length chunks or blocks"
            # chunks('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return zip(*args)

        data = zip(input_data, target_data)

        for epoch in xrange(epochs):

            if show_progress: print("Epoch %d/%d" % (epoch + 1, epochs))

            random.shuffle(data)

            for batch in chunks(data, batch_size):
                deltas = []
                # errors = np.zeros(batch_size)

                for i, couple in enumerate(batch):
                    if couple is None:
                        continue

                    x, target = couple
                    y = self.forward(x)
                    J = loss.calc_loss(y, target)
                    dJdy = loss.calc_gradient(y, target)

                    deltas.append(dJdy)

                    # if save_progress:
                    #     errors[i] = np.sum(J)

                # Sum all the dJdy vectors "element-wise"
                # print('deltas', deltas)
                mean_delta = np.sum(deltas, axis=0) / float(batch_size)

                # mean_error = np.mean(errors)
                # print('mean_delta', mean_delta)
                self.backward(mean_delta, optimizer)

                # if save_progress:
                #     self.errors_history.append(mean_error)
                #     self.loss_gradient_history.append(np.mean(mean_grad))

    def test(self, input_data, target_data, loss):
        for x, target in izip(input_data, target_data):
            y = self.forward(x)
            print (y)
            output = loss.output(y)
            print output, target


    def test_analysis(self, input_data, target_data):
        pass

    def save_to_file(self, file_name):
        print('Saving model to file...')
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
            print('Saved.')

    @staticmethod
    def load_from_file(file_name):
        print('Loading model from file...')
        with open(file_name, 'r') as f:
            model = pickle.load(f)
            print('Loaded')
            return model

    def show_progress(self, show_progress, count, input_data):
        if show_progress:
            if count % 1000 == 0:
                progress = (count * 100) / len(input_data)
                print("Training %d%% (%d of %d)" % (progress, count, len(input_data)))

    def plot_errors_history(self):
        plt.figure()
        plt.title('Errors History (J)')
        plt.plot(xrange(len(self.errors_history)), self.errors_history, color='red')
        plt.ylim([0, 2])
        return plt

    def plot_loss_gradient_history(self):
        plt.figure()
        plt.title('Loss Gradient History (dJ/dy)')
        plt.plot(xrange(len(self.loss_gradient_history)), self.loss_gradient_history, color='orange')
        plt.ylim([0, 2])
        return plt
