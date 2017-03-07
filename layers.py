import numpy as np


class LinearLayer:
    def __init__(self, inputs, outputs, initialize="random"):
        self.inputs = inputs
        self.outputs = outputs
        self.x = None

        if initialize == 'random':
            self.W = np.random.rand(outputs, inputs + 1)
        elif initialize == 'ones':
            self.W = np.ones([outputs, inputs + 1])
        elif initialize == 'zeros':
            self.W = np.zeros([outputs, inputs + 1])
        else:
            raise Exception("Unrecognized initialization value")

    def forward(self, x, is_training=False):
        self.vector_type_check(x)
        x = np.hstack([1, x])
        self.x = x
        y = self.W.dot(x)
        return y

    def backward(self, dJdy):
        weights_without_bias = self.W[:, 1:]
        return weights_without_bias.T.dot(dJdy)

    def update_gradient(self, dJdy):
        grad = np.multiply(np.matrix(self.x).T, dJdy).T
        # self.W += grad
        return grad

    @staticmethod
    def vector_type_check(x):
        if x.dtype == 'int64':
            raise Exception("input is int64 type. It should be float")


class SigmoidLayer:
    def __init__(self):
        pass

    def forward(self, x, is_training=False):
        assert x.ndim == 1, "Sigmoid input is not one-dimensional"
        self.sigm_x = 1. / (1. + np.exp(-x))
        return self.sigm_x

    def backward(self, dJdy):
        dydx = (1. - self.sigm_x) * self.sigm_x
        return dydx * dJdy


class SignLayer:
    def forward(self, x, is_training=False):
        return np.sign(x)

    def backward(self, dJdy):
        return dJdy


class ReluLayer:
    def forward(self, x, is_training=False):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dJdy):
        # self.x >= 0 returns a vector with True/False booleans.
        # When multiplied by a 1. scalar you get a 1/0 bitmask
        bitmask = (self.x >= 0) * 1
        return bitmask * dJdy


class TanhLayer:
    def forward(self, x, is_training=False):
        self.y = np.tanh(x)
        return np.tanh(x)

    def backward(self, dJdy):
        return (1. - self.y ** 2) * dJdy


class SoftmaxLayer:
    def forward(self, x, is_training=False):
        exp_x = np.exp(x)
        self.y = exp_x / np.sum(exp_x)
        return self.y

    def backward(self, dJdy):
        y_rows = self.y.reshape((-1, 1))
        y_squared_matrix = y_rows.dot(y_rows.T)
        y_eyed = np.eye(self.y.size) * self.y
        dxdy = y_eyed - y_squared_matrix
        out = dJdy * dxdy
        return np.sum(out, axis=1)


class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.binomial = None

    def forward(self, x, is_training=False):
        if is_training:
            if self.binomial is None:
                self.binomial = np.random.binomial(1, self.p, size=x.shape)
            return self.binomial * x
        else:
            return x

    def backward(self, dJdy):
        return dJdy


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

    def forward(self, x, is_training=False):
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
