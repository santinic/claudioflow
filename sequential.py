import matplotlib.pyplot as plt
from itertools import izip
import cPickle as pickle


class GenericModel:
    def validate_input_data(self, x):
        if type(x) == list:
            raise Exception("Input shouldn't be a Python list, but a numpy.ndarray.")

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


class SequentialModel(GenericModel):
    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers
        self.errors_history = []
        self.loss_gradient_history = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x, is_training=False):
        self.validate_input_data(x)

        y = None
        for layer in self.layers:
            y = layer.forward(x, is_training)
            x = y
        return y

    def forward_all(self, xs, is_training=False):
        return map(lambda x: self.forward(x, is_training), xs)

    def backward(self, delta, optimizer=None, update=True, backward_first_layer=True):
        for i, layer in reversed(list(enumerate(self.layers))):
            jump_first_layer = (i == 0) and not backward_first_layer

            if not jump_first_layer:
                delta_for_previous_layer = layer.backward(delta)

            # If the layer has some internal state we update it with the optimizer
            if hasattr(layer, 'update_gradient') and update and (optimizer is not None):
                update_gradient = layer.update_gradient(delta)
                optimizer.update(layer, update_gradient)

            if not jump_first_layer:
                delta = delta_for_previous_layer
        return delta

