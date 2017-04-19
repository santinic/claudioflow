import numpy as np

from network import Layer
from utils import split_array_into_variable_sizes


class Linear(object):
    def __init__(self, in_size, out_size, initialize="random", dtype=None):
        self.x = None
        self.dtype = dtype
        self.first_x_already_checked = False

        self.delta_W = np.zeros([out_size, in_size + 1], dtype=dtype)

        if initialize == 'random':
            self.W = np.random.rand(out_size, in_size + 1).astype(dtype)
        elif initialize == 'ones':
            self.W = np.ones([out_size, in_size + 1], dtype=dtype)
        elif initialize == 'zeros':
            self.W = np.zeros([out_size, in_size + 1], dtype=dtype)
        else:
            raise Exception("Unrecognized initialization value")

    def forward(self, x, is_training=False):
        self.check_first_x_dtype(x)
        x = np.hstack([1., x])
        self.x = x
        y = self.W.dot(x)
        return y

    def backward(self, dJdy):
        self.delta_W += self.calc_update_gradient(dJdy)
        weights_without_bias = self.W[:, 1:]
        return weights_without_bias.T.dot(dJdy)

    def calc_update_gradient(self, dJdy):
        grad = np.multiply(np.matrix(self.x).T, dJdy).T
        return grad

    def update_weights(self, optimizer):
        optimizer.update(self, self.delta_W)
        # reset delta_W
        self.delta_W = np.zeros(self.W.shape)

    def check_first_x_dtype(self, x):
        if self.first_x_already_checked:
            return
        if x.dtype == 'int64':
            raise Exception("input is int64 type. It should be float")
        if self.dtype is not None:
            if self.dtype != x.dtype:
                raise Exception("input has dtype=%s, while LinearLayer configured as dtype=%s" % (x.dtype, self.dtype))
        self.first_x_already_checked = True


class RegularizedLinear(Linear):
    def __init__(self, n_inputs, out_size, initialize='random', l1=0., l2=0.):
        super(RegularizedLinear, self).__init__(n_inputs, out_size, initialize)
        self.l1 = l1
        self.l2 = l2

    def calc_update_gradient(self, dJdy):
        grad = super(RegularizedLinear, self).calc_update_gradient(dJdy)
        l1_reg = self.l1 * np.sign(self.W)
        l2_reg = self.l2 * self.W
        return grad + l1_reg + l2_reg


class Wx(Linear):
    def __init__(self, in_size, out_size, initialize='random'):
        if initialize == 'random':
            self.W = np.random.rand(out_size, in_size)
        elif initialize == 'ones':
            self.W = np.ones([out_size, in_size])
        elif initialize == 'zeros':
            self.W = np.zeros([out_size, in_size])

        self.delta_W = np.zeros(self.W.shape)

    def forward(self, x, is_training=False):
        self.x = x
        return self.W.dot(x)

    def backward(self, dJdy):
        self.delta_W += self.calc_update_gradient(dJdy)
        return self.W.T.dot(dJdy)

    def update_weights(self, optimizer):
        optimizer.update(self, self.delta_W)
        self.delta_W = np.zeros(self.W.shape)


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, x, is_training=False):
        self.sigm_x = 1. / (1. + np.exp(-x))
        return self.sigm_x

    def backward(self, dJdy):
        dydx = (1. - self.sigm_x) * self.sigm_x
        return dydx * dJdy


class Sign(Layer):
    def forward(self, x, is_training=False):
        return np.sign(x)

    def backward(self, dJdy):
        return dJdy


class Relu(Layer):
    def forward(self, x, is_training=False):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dJdy):
        # self.x >= 0 returns a vector with True/False booleans.
        # When multiplied by a 1. scalar you get a 1/0 bitmask
        bitmask = (self.x >= 0)
        return bitmask * dJdy


class Tanh(Layer):
    def forward(self, x, is_training=False):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dJdy):
        return (1. - self.y ** 2) * dJdy


class CheapTanh(Layer):
    def __init__(self, alpha=1.):
        self.alpha = alpha

    def forward(self, x, is_training=False):
        self.x = x
        a = self.alpha
        y = np.where(x <= -a, -a, x)
        y = np.where(y >= +a, +a, y)
        return y

    def backward(self, dJdy):
        a = self.alpha
        grad = np.array([1. if -a < x < a else 0. for x in self.x])
        return grad * dJdy


# class CheapSigmoidLayer:
#     def __init__(self, alpha=1.):
#         self.alpha = alpha
#
#     def forward(self, x, is_training=False):
#         self.x = x
#         a = self.alpha
#         y = np.where(x <= -a, 0, x)
#         y = np.where(y >= +a, 1, y)
#         return y
#
#     def backward(self, dJdy):
#         a = self.alpha
#         grad = np.array([1. if -a < x < a else 0. for x in self.x])
#         return grad * dJdy


class Softmax(Layer):
    def forward(self, x, is_training=False):
        c = np.max(x)
        exp_x = np.exp(x - c)
        self.y = exp_x / np.sum(exp_x)
        return self.y

    def backward(self, dJdy):
        y_rows = self.y.reshape((-1, 1))
        y_squared_matrix = y_rows.dot(y_rows.T)
        y_eyed = np.eye(self.y.size) * self.y
        dxdy = y_eyed - y_squared_matrix
        out = dJdy * dxdy
        return np.sum(out, axis=1)

        # def backward(self, dJdy):
        #     dJdx = np.zeros(dJdy.size)
        #     for i in range(self.y.size):
        #         aux_y = -self.y.copy()
        #         aux_y[i] = (1-self.y[i])
        #         dJdx[i] = self.y[i]*aux_y.dot(dJdy)
        #     return dJdx


class Dropout(Layer):
    def __init__(self, p):
        self.p = p
        self.binomial = None

    def forward(self, x, is_training=False):
        if is_training:
            if self.binomial is None:
                self.binomial = np.random.binomial(1, self.p, size=x.shape)
            return self.binomial * x
        else:
            return x * self.p

    def backward(self, dJdy):
        ret = dJdy * self.binomial
        self.binomial = None
        return ret


class ClaMax(Layer):
    def forward(self, x):
        self.sum_x = np.sum(x)
        self.y = x / self.sum_x
        return self.y

    def backward(self, err):
        y_eyed = np.eye(self.y.size) * (1. / self.sum_x - self.y)
        return y_eyed


class Sum(Layer):
    '''
    Sum Layer: [a,b,c] => a+b+c
    backward: [dJdy, dJdy, dJdy]
    '''

    def forward(self, x, is_training=False):
        self.size_of_x = x.size
        y = np.sum(x, axis=0)
        self.size_of_y = y.size
        return y

    def backward(self, dJdy):
        return dJdy * np.ones(self.size_of_y)


class Mul(Layer):
    """
    Multiplication Layer: [a,b,c] => a*b*c
    backward: dJdy * [bc, ac, ab]
    """

    def forward(self, x, is_training=False):
        self.x = x
        self.y = np.prod(x, axis=0)
        return self.y

    def backward(self, dJdy):
        return dJdy * (self.y / self.x)


class Const(Layer):
    def __init__(self, const=1.):
        self.const = np.array(const)

    def forward(self, x, is_training=False):
        return self.const

    def backward(self, dJdy):
        return np.array([0])


class Store(Layer):
    def __init__(self, in_size):
        self.x = np.zeros(in_size)

    def forward(self, x, is_training=False):
        self.x = x
        return x

    def backward(self, dJdy):
        self.grad = dJdy
        return dJdy

    def read_forward(self):
        return self.x

    # def read_backward(self):
    #     return self.grad


class Concat(Layer):
    def forward(self, values):
        self.sizes = map(len, values)
        return np.hstack(values)

    def backward(self, dJdy):
        return split_array_into_variable_sizes(dJdy, self.sizes)


class Print(Layer):
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
