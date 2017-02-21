from itertools import izip

import numpy as np
import matplotlib.pyplot as plt


class SequentialModel:

    def __init__(self, input_size, layers=None):
        self.layers = []
        self.input_size = input_size

        self.errors_history = []
        self.loss_gradient_history = []

        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self.layers.append(layer)

    def validate_input_data(self, x):
        if type(x) == list:
            raise Exception("Input shouldn't be a Python list, but a numpy.ndarray.")
        if x.size != self.input_size:
            raise Exception("Input size should be %d, is instead %d" % (self.input_size, x.size))

    def forward(self, x):
        self.validate_input_data(x)

        y = None
        for i, layer in enumerate(self.layers):
            y = layer.forward(x)
            x = y
        return y

    def backward(self, dJdy, update=True):
        for layer in reversed(self.layers):
            new_dJdy = layer.backward(dJdy)

            # If the layer has some internal state we update it
            if hasattr(layer, 'update') and update:
                # It's important the update() gets called *after* backward()
                layer.update(dJdy)
            dJdy = new_dJdy

        return dJdy

    def learn_one(self, x, target, loss, learning_rate, show_progress=False):
        y = self.forward(x)
        J = loss.calc_loss(y, target)
        dJdy = loss.calc_gradient(y, target)
        self.backward(learning_rate * dJdy)
        return J, dJdy

    def learn(self, input_data, target_data, loss, epochs, learning_rate, show_progress=False):
        self.errors_history = []
        for epoch in xrange(epochs):
            if show_progress: print("Epoch %d/%d" % (epoch+1, epochs))
            count = 0
            for x, target in izip(input_data, target_data):
                self.show_progress(show_progress, count, input_data)
                J, dJdy = self.learn_one(x, target, loss, learning_rate, show_progress=show_progress)
                self.errors_history.append(J)
                self.loss_gradient_history.append(dJdy)
                count += 1
        return self.errors_history

    def show_progress(self, show_progress, count, input_data):
        if show_progress:
            if count % 1000 == 0:
                progress = (count * 100) / len(input_data)
                print("Training %d%% (%d of %d)" % (progress, count, len(input_data)))

    def plot_errors_history(self):
        plt.figure()
        plt.title('Errors History (J)')
        plt.plot(xrange(len(self.errors_history)), self.errors_history, color='red')
        plt.ylim([0,2])
        return plt

    def plot_loss_gradient_history(self):
        plt.figure()
        plt.title('Loss Gradient History (dJ/dy)')
        plt.plot(xrange(len(self.loss_gradient_history)), self.loss_gradient_history, color='orange')
        plt.ylim([0,2])
        return plt

class LinearLayerVectorial:

    def __init__(self, inputs, outputs, initialize="random"):
        self.inputs = inputs
        self.outputs = outputs
        self.x = None

        if initialize == 'random':
            self.W = np.random.rand(outputs, inputs+1)
        elif initialize == 'ones':
            self.W = np.ones([outputs, inputs+1])
        else:
            raise Exception("Unrecognized initialization value")

    def forward(self, x):
        self.vector_type_check(x)
        x = np.hstack([1, x])
        self.x = x
        y = self.W.dot(x)
        return y

    def backward(self, dJdy):
        weights_without_bias = self.W[ : ,1: ]
        out = weights_without_bias.T.dot(dJdy)
        return out

    def update(self, dJdy):
        # print('======')
        # print('W', self.W)
        # print('dJdy', dJdy)
        # print('x', self.x)
        # print('delta', np.multiply(np.matrix(self.x).T, dJdy).T)
        self.W += np.multiply(np.matrix(self.x).T, dJdy).T

    def vector_type_check(self, x):
        if x.dtype == 'int64':
            raise Exception("input is int64 type. It should be float")


LinearLayer = LinearLayerVectorial


class SigmoidLayer:

    def __init__(self):
        pass

    def forward(self, x):
        assert x.ndim == 1, "Sigmoid input is not one-dimensional"
        self.sigm_x = 1. / (1. + np.exp(-x))
        return self.sigm_x

    def backward(self, dJdy):
        dydx = (1. - self.sigm_x) * self.sigm_x
        out = dydx * dJdy
        return out


class SignLayer:

    def forward(self, x):
        return np.sign(x)

    def backward(self, dJdy):
        return dJdy


class ReluLayer:

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dJdy):
        # self.x >= 0 returns a vector with True/False booleans.
        # When multiplied by a 1. scalar you get a 1/0 bitmask
        bitmask = (self.x >= 0) * 1
        return bitmask * dJdy


class SoftmaxLayer:

    def forward(self, x):
        exp_x = np.exp(x)
        self.y = exp_x / np.sum(exp_x)
        return self.y

    def backward(self, dJdy):
        y_rows = self.y.reshape((-1, 1))
        y_squared_matrix = y_rows.dot(y_rows.T)
        y_eyed = np.eye(self.y.size) * self.y
        dxdy = y_eyed - y_squared_matrix
        # print("softmax backward:", dJdy * dxdy)
        out = dJdy * dxdy
        return np.sum(out, axis=1)


# class ClaMaxLayer:
#
#     def forward(self, x):
#         self.sum_x = np.sum(x)
#         self.y = x / self.sum_x
#         return self.y
#
#     def backward(self, err):
#         y_eyed = np.eye(self.y.size) * (1./self.sum_x - self.y)
#         return y_eyed


class PrintLayer:

    def __init__(self, prefix=None):
        self.prefix = prefix

    def forward(self, x):
        if self.prefix is not None:
            print('=> %s:\n%s' % (self.prefix, x))
        else:
            print(x)
        return x

    def backward(self, dJdy):
        if self.prefix:
            print('<= %s:\n%s' % (self.prefix, dJdy))
        else:
            print(dJdy)
        return dJdy


class SquaredLoss:

    def calc_loss(self, y, target):
        J = 0.5 * (target - y)**2
        return J

    def calc_gradient(self, y, target):
        dJdy = - (y - target)
        return dJdy


class NegLogLikelihoodLoss:

    def calc_loss(self, y, target):
        target = int(target)
        J = - np.log(y[target])
        return J

    def calc_gradient(self, y, target):
        dJdy = - y
        dJdy[target] += 1.
        return dJdy

NLL = NegLogLikelihoodLoss
