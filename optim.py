import numpy as np

"""
Optimization methods implemented as described in the Deep Learning Book:
http://www.deeplearningbook.org/contents/optimization.html
"""


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer, grad):
        layer.W -= self.learning_rate * grad


class MomentumSGD:
    def __init__(self, learning_rate, momentum=0.):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer, grad):
        if not hasattr(layer, 'velocity'):
            layer.velocity = 0.
        layer.velocity = (self.momentum * layer.velocity) - (self.learning_rate * grad)
        layer.W += layer.velocity


class AdaGrad:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.delta_const = 1e-8

    def update(self, layer, grad):
        if not hasattr(layer, 'r'):
            layer.r = np.zeros(grad.shape)
        layer.r += grad * grad
        layer.W += - (self.learning_rate * grad) / np.sqrt(self.delta_const + layer.r)


class RMSProp:
    def __init__(self, learning_rate, decay_rate):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.delta_const = 10 ** -6

    def update(self, layer, grad):
        if not hasattr(layer, 'r'):
            layer.r = np.zeros(grad.shape)
        squared_grad = np.multiply(grad, grad)
        layer.r = (self.decay_rate * layer.r) + (1. - self.decay_rate) * squared_grad
        left = - self.learning_rate / np.sqrt(self.delta_const + layer.r)
        layer.W += np.multiply(left, grad)
