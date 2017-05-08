import numpy as np

from network import Layer, Seq
from utils import split_array_into_variable_sizes


class Linear(object):
    def __init__(self, in_size, out_size, initialize="random", dtype=None):
        self.x = None
        self.dtype = dtype
        self.first_x_already_checked = False

        # self.delta_W = np.zeros([out_size, in_size + 1], dtype=dtype)

        if isinstance(initialize, str):
            self.W = MatrixWeight(in_size, out_size, initialize)
        elif isinstance(initialize, MatrixWeight):
            self.W = initialize
        # if type(initialize) is not str:
        #     self.W = initialize
        # elif initialize == 'random':
        #     self.W = np.random.rand(out_size, in_size + 1).astype(dtype)
        # elif initialize == 'randn':
        #     self.W = np.random.randn(out_size, in_size + 1)
        # elif initialize == 'ones':
        #     self.W = np.ones([out_size, in_size + 1], dtype=dtype)
        # elif initialize == 'zeros':
        #     self.W = np.zeros([out_size, in_size + 1], dtype=dtype)
        else:
            raise Exception("Unrecognized initialization value")

    def forward(self, x, is_training=False):
        self.check_first_x_dtype(x)
        x = np.hstack([1., x])
        self.x = x
        y = self.W.get().dot(x)
        return y

    def backward(self, dJdy):
        self.W.delta += self.calc_update_gradient(dJdy)
        weights_without_bias = self.W.get()[:, 1:]
        return weights_without_bias.T.dot(dJdy)

    def calc_update_gradient(self, dJdy):
        grad = np.multiply(np.matrix(self.x).T, dJdy).T
        return grad

    def update_weights(self, optimizer):
        optimizer.update(self.W, self.W.delta)
        self.W.delta.fill(0.)

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


class MatrixWeight:
    def __init__(self, in_size=None, out_size=None, initialize=None, scale=1):
        assert type(initialize) == str or type(initialize) == np.ndarray or \
               type(initialize) == np.matrixlib.defmatrix.matrix, \
            "%s" % type(initialize)

        if type(initialize) is not str:
            self.W = initialize
        elif initialize == 'random':
            self.W = np.random.rand(out_size, in_size) * scale
        elif initialize == 'randn':
            self.W = np.random.randn(out_size, in_size) * scale
        elif initialize == 'ones':
            self.W = np.ones([out_size, in_size]) * scale
        elif initialize == 'zeros':
            self.W = np.zeros([out_size, in_size])
        else:
            raise Exception("Unrecognized initialization value")

        assert isinstance(self.W, np.ndarray)
        self.delta = np.zeros_like(self.W)

    def get(self):
        return self.W

    def get_delta(self):
        return self.delta


class Wx(Linear):
    def __init__(self, in_size, out_size, initialize='random'):
        if isinstance(initialize, MatrixWeight):
            self.W = initialize
        elif isinstance(initialize, str):
            self.W = MatrixWeight(in_size, out_size, initialize)
        else:
            raise Exception("Unrecognized initialization value")

        # print('Wx initialized with', self.W.get())

    def forward(self, x, is_training=False):
        self.x = x
        return self.W.get().dot(x)

    def backward(self, dJdy):
        self.W.delta += self.calc_update_gradient(dJdy)
        return self.W.get().T.dot(dJdy)

    def update_weights(self, optimizer):
        optimizer.update(self.W, self.W.delta)
        self.W.delta.fill(0.)


class VectorWeight:
    def __init__(self, in_size, initialize, scale=1):
        assert type(in_size) == int

        if type(initialize) is not str:
            self.W = initialize
        elif initialize == 'random':
            self.W = np.random.rand(in_size) * scale
        elif initialize == 'randn':
            self.W = np.random.randn(in_size) * scale
        elif initialize == 'ones':
            self.W = np.ones(in_size) * scale
        elif initialize == 'zeros':
            self.W = np.zeros(in_size)
        else:
            raise Exception("Unrecognized initialization value")

        self.delta = np.zeros_like(self.W)

    def get(self):
        return self.W

    def get_delta(self):
        return self.delta


class PlusBias(Layer):
    def __init__(self, in_size, initialize='random'):
        if isinstance(initialize, VectorWeight):
            self.b = initialize
        else:
            self.b = VectorWeight(in_size, initialize)

    def forward(self, x, is_training=False):
        return x + self.b.get()

    def backward(self, dJdy):
        self.b.delta += dJdy
        return dJdy

    def update_weights(self, optimizer):
        optimizer.update(self.b, self.b.delta)
        self.b.delta.fill(0.)


class WxBiasLinear(Layer):
    def __init__(self, in_size, out_size, initialize_W, initialize_b):
        self.Wx = Wx(in_size, out_size, initialize_W)
        self.bias = PlusBias(out_size, initialize_b)
        self.model = Seq(self.Wx, self.bias)

    def forward(self, x, is_training=False):
        return self.model.forward(x, is_training)

    def backward(self, dJdy):
        return self.model.backward(dJdy)

    def update_weights(self, optimizer):
        return self.model.update_weights(optimizer)


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
        # print('-> Tanh %d registers' % id(self), self.y)
        return self.y

    def backward(self, dJdy):
        # print('<- Tanh %d backward with ' % id(self), self.y, dJdy)
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
        return ret

    def update_weights(self, optimizer):
        self.binomial = None


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

    def forward(self, xs, is_training=False):
        self.elements = xs.shape[0]
        # self.vector_size = xs.shape[1]
        y = np.sum(xs, axis=0)
        return y

    def backward(self, dJdy):
        # assert dJdy.shape[0] == self.vector_size, "backward dJdy size is not compatible with previous x vector size"
        return [dJdy] * self.elements


class Neg(Layer):
    def forward(self, x, is_training=False):
        return -x

    def backward(self, dJdy):
        return -dJdy


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
    def forward(self, values, is_training=False):
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


class SyntaxLayer(Layer):
    def __init__(self, expr):
        self.model = expr

    def forward(self, x, is_training=False):
        return self.model.forward_variables({'x': x})

    def backward(self, dJdy):
        return self.model.backward_variables(dJdy)['x']

    def update_weights(self, optimizer):
        return self.model.update_weights(optimizer)
