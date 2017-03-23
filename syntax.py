import unittest

import numpy as np
from numpy.testing import assert_array_equal

import layers


class Op:
    def __init__(self, input=None):
        self.inputs = [input]

    def __add__(self, other):
        return Sum(self, other)

    def __sub__(self, other):
        raise Exception("Subtraction not implemented")
        # return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __div__(self, other):
        raise Exception("Division not implemented")
        # return Div(self, other)

    def __str__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        input_str = ''
        if len(self.inputs) > 0:
            input_str = ', '.join([str(x) for x in self.inputs])
            input_str = '(%s)' % input_str
        return '%s%s' % (name, input_str)

    def explore_forward(self, input_dict, depth=0):
        values = []
        for input_op in self.inputs:
            val = input_op.explore_forward(input_dict, depth + 1)
            values.append(val)
            print '%s%s -> %s' % ('\t' * depth, input_op, val)
        x = values[0] if len(values) == 1 else values
        y = self.forward(np.array(x))
        print("forward of", self, self.inputs, y)
        return y

    def explore_backward(self, deltas, depth=0):
        print '%s%s <- deltas=%s' % ('\t' * depth, self, deltas)
        deltas = self.backward(deltas)
        if len(self.inputs) == 1:
            return self.inputs[0].explore_backward(deltas, depth=depth + 1)
        else:
            ret_deltas = []
            for input_op, dJdy in zip(self.inputs, deltas):
                delta = input_op.explore_backward(dJdy, depth=depth + 1)
                ret_deltas.append(delta)
        # print('\t'*depth+'unmerged'+str(ret_deltas))
        merged = self.merge_backprop_dicts(ret_deltas)
        # print('\t'*depth+'merged'+str(merged))
        return merged

    @staticmethod
    def merge_backprop_dicts(dicts):
        merged = dicts[0]
        for dict in dicts[1:]:
            for k, v in dict.iteritems():
                if k in merged:
                    merged[k] += v
                else:
                    merged[k] = v
        return merged

    def forward(self, values):
        return self.layer.forward(values)

    def backward(self, dJdy):
        return self.layer.backward(dJdy)


class Var(Op):
    def __init__(self, variable_name):
        self.variable_name = variable_name
        self.inputs = []

    def explore_forward(self, input_dict, depth):
        val = input_dict[self.variable_name]
        return val

    def explore_backward(self, deltas, depth):
        grad_dict = {}
        grad_dict[self.variable_name] = deltas
        return grad_dict

    def forward(self, values):
        return values

    def backward(self, dJdy):
        return dJdy

    def __str__(self):
        return Op.__str__(self, name=self.variable_name)


class Linear(Op):
    def __init__(self, in_size, out_size, inputs=[]):
        Op.__init__(self, inputs)
        self.layer = layers.Linear(in_size, out_size)


class Tanh(Op):
    layer = layers.Tanh()


class Sigmoid(Op):
    layer = layers.Sigmoid()


class Sum(Op):
    def __init__(self, *args):
        self.layer = layers.Sum()
        self.inputs = args


class Mul(Op):
    def __init__(self, *args):
        self.layer = layers.Mul()
        self.inputs = args


class Const(Op):
    def __init__(self, const):
        Op.__init__(self)
        self.layer = layers.Const(const)
        self.inputs = []


class SyntaxTest(unittest.TestCase):
    def test_merge_dicts(self):
        dict = [
            {'a': np.array([1, 1])},
            {'a': np.array([2, 2])},
            {'b': np.array([4, 4])}
        ]
        merged = Op.merge_backprop_dicts(dict)
        assert_array_equal(merged['a'], np.array([3, 3]))
        assert_array_equal(merged['b'], np.array([4, 4]))

    def test_basic(self):
        a_var = Var('a')
        b_var = Var('b')
        a = np.array([2., 3.])
        b = np.array([3., 5.])
        input_dict = {'a': a, 'b': b}

        # 2a+b
        model = (a_var + b_var + a_var)
        y = model.explore_forward(input_dict)
        assert_array_equal(y, a + a + b)
        grad = model.explore_backward(np.ones(2))
        assert_array_equal(grad['a'], [2])
        assert_array_equal(grad['b'], [1])

        # 2a*b
        model = a_var * b_var * a_var

        y = model.explore_forward({'a': a, 'b': b})
        assert_array_equal(y, a * a * b)

        grad = model.explore_backward(np.ones(2))
        assert_array_equal(grad['a'], 2 * a * b)
        assert_array_equal(grad['b'], a * a)

    def test_just_numbers(self):
        a = Var('a')
        b = Var('b')
        input_dict = {'a': [2.], 'b': [3.]}

        model = (a + b)
        y = model.explore_forward(input_dict)
        assert_array_equal(y, [5])

        model = (a * b)
        y = model.explore_forward(input_dict)
        assert_array_equal(y, [6])

    def test_syntax(self):
        var_a = Var('a')
        var_b = Var('b')
        a = np.array([2.3, 3., 3])
        b = np.array([3., 5., 4])
        input_dict = {'a': a, 'b': b}

        model = Sigmoid(var_a * var_a) * (var_b * var_b * var_b + Const(2) * var_a)

        y = model.explore_forward(input_dict)
        assert_array_equal(y, Sigmoid.forward(a ** 2) * (b ** 3 + 2 * a))

        grad = model.explore_backward(np.ones(3))
        assert_array_equal(grad['a'], 2 * Sigmoid.forward(a) * (1 - a) + 2)
        assert_array_equal(grad['b'], 3 * b * b)
