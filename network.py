import inspect

import numpy as np
import cPickle as pickle


class Layer:
    def update_weights(self, optimizer):
        return

    def forward_all(self, xs, is_training=False):
        return map(lambda x: self.forward(x, is_training), xs)


class Serializable:
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


class WithLayers:
    def __init__(self, *args):
        self.layers = []
        if len(args) == 1 and type(args[0]) == list:
            args = args[0]
        for layer in args:
            self.add(layer)

    def add(self, layer):
        if inspect.isclass(layer):
            # instantiate class
            layer = layer()
        self.layers.append(layer)


class Seq(Layer, WithLayers, Serializable):
    def forward(self, x, is_training=False):
        self.validate_input_data(x)

        y = None
        for layer in self.layers:
            y = layer.forward(x, is_training)
            x = y
        return y

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def update_weights(self, optimizer):
        for layer in reversed(self.layers):
            delta = layer.update_weights(optimizer)
        return delta

    def validate_input_data(self, x):
        if type(x) == list:
            raise Exception("Input shouldn't be a Python list, but a numpy.ndarray.")


class Par(Layer, WithLayers, Serializable):
    def forward(self, xs, is_training=False):
        ys = []
        for layer in self.layers:
            y = layer.forward(xs, is_training)
            ys.append(y)
        return np.array(ys)

    def backward(self, dJdy):
        dJdys = []
        for layer in self.layers:
            dJdys.append(layer.backward(dJdy))
        return np.array(dJdys)

    def backward_and_update(self, dJdy, optimizer):
        dJdys = []
        for layer in self.layers:
            dJdys.append(layer.backward_and_update(dJdy, optimizer))
        return np.array(dJdys)


class Map(Layer, WithLayers, Serializable):
    def forward(self, xs, is_training=False):
        y = map(lambda (layer, x): layer.forward(x), zip(self.layers, xs))
        return np.array(y)

    def backward(self, grads):
        back = map(lambda (layer, dJdy): layer.backward(dJdy), zip(self.layers, grads))
        return np.array(back)

    def backward_and_update(self, grads, optimizer):
        back = map(lambda (layer, dJdy): layer.backward_and_update(dJdy, optimizer), zip(self.layers, grads))
        return np.array(back)


class Identity(Layer):
    def forward(self, xs, is_training=False):
        return xs

    def backward(self, dJdy):
        return dJdy

# class Sum(ModelWithLayers):
#     def __init__(self, *args):
#         self.model = Seq(Par(*args), layers.SumLayer)
#
#     def forward(self, xs, is_training=False):
#         return self.model.forward(xs, is_training)
#
#     def backward(self, dJdy):
#         return self.model.backward(dJdy)


# class Mul(ModelWithLayers):
#     def __init__(self, *args):
#         self.model = Seq(Par(*args), layers.MulLayer)
#
#     def forward(self, xs, is_training=False):
#         return self.model.forward(xs, is_training)
#
#     def backward(self, dJdy):
#         return self.model.backward(dJdy)
