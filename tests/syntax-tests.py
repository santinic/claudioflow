import unittest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

import layers
from optim import SGD
from syntax import Var, Concat, Linear, Store, Sigmoid, SyntaxOp


class SyntaxTest(unittest.TestCase):
    def test_merge_dicts(self):
        dict = [
            {'a': np.array([1, 1])},
            {'a': np.array([2, 2])},
            {'b': np.array([4, 4])}
        ]
        merged = SyntaxOp.merge_backprop_dicts(dict)
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
        y = model.forward_variables(input_dict)
        assert_array_equal(y, a + a + b)
        grad = model.backward_variables(np.ones(2))
        assert_array_equal(grad['a'], [2])
        assert_array_equal(grad['b'], [1])

        # 2a*b
        model = a_var * b_var * a_var

        y = model.forward_variables({'a': a, 'b': b})
        assert_array_equal(y, a * a * b)

        grad = model.backward_variables(np.ones(2))
        assert_array_equal(grad['a'], 2 * a * b)
        assert_array_equal(grad['b'], a * a)

    def test_just_numbers(self):
        a = Var('a')
        b = Var('b')
        input_dict = {'a': [2.], 'b': [3.]}

        model = (a + b)
        y = model.forward_variables(input_dict)
        assert_array_equal(y, [5])

        model = (a * b)
        y = model.forward_variables(input_dict)
        assert_array_equal(y, [6])

    def test_proper_equation(self):
        var_a = Var('a')
        var_b = Var('b')
        a = np.array([2.3, 3., 3])
        b = np.array([3., 5., 4])
        input_dict = {'a': a, 'b': b}
        sigm = layers.Sigmoid().forward

        model = Sigmoid(var_a * var_a) + var_b * var_b * var_b + var_a
        print(model)

        y = model.forward_variables(input_dict)
        assert_array_equal(y, sigm(a ** 2) + (b ** 3 + a))

        grad = model.backward_variables(np.ones(3))
        assert_array_equal(grad['a'], 2 * a * sigm(a ** 2) * (1 - sigm(a ** 2)) + 1)
        assert_array_equal(grad['b'], 3 * b * b)

    def test_compare_linear_syntax_and_linear_layer(self):
        x = np.random.rand(3)
        syntax_model = Linear(3, 4, initialize='ones', input=Var('x'))
        layer_model = layers.Linear(3, 4, initialize='ones')

        syntax_y = syntax_model.forward_variables({'x': x})
        layer_y = layer_model.forward(x)
        assert_array_equal(syntax_y, layer_y)

        dJdy = np.random.rand(4)
        syntax_grad = syntax_model.backward_variables(dJdy)
        layer_grad = layer_model.backward(dJdy)
        self.assertEqual(syntax_grad['x'].shape, layer_grad.shape, 'gradients should have the same vector shape')
        assert_array_almost_equal(syntax_grad['x'], layer_grad)


    def test_linear_and_store(self):
        var_a = Var('a')
        var_b = Var('b')
        a = np.array([2.3, 3., 3])
        b = np.array([3., 5., 4])

        stored_ab = Store(3, var_a * var_b)
        linear = Linear(3, 3, initialize='ones', input=stored_ab)
        model = var_a + linear + var_a * var_b

        y = model.forward_variables({'a': a, 'b': b})
        assert_array_equal(y, np.sum(a * b) + 1 + a + a * b)

        grad = model.backward_variables(np.ones(1))
        assert_array_equal(grad['a'], b + 1 + b)
        assert_array_equal(grad['b'], a + a)

        assert_array_equal(stored_ab.read_forward(), a * b)

    def test_update_weights(self):
        x = np.array([1., 2., 3.])
        optimizer = SGD(0.1)
        linear_layer_model = layers.Linear(3, 3, initialize='ones')
        y = linear_layer_model.forward(x)
        linear_layer_model.backward(np.ones(3))
        W_before_update1 = linear_layer_model.W.copy()
        linear_layer_model.update_weights(optimizer)

        var_x = Var('x')
        syntax_model = Linear(3, 3, initialize='ones', input=var_x)
        syntax_model.forward_variables({'x': x})
        syntax_model.backward_variables(np.ones(3))
        W_before_update2 = syntax_model.layer.W
        syntax_model.update_weights(optimizer)

        print(W_before_update1, W_before_update2)
        assert_array_equal(linear_layer_model.W, syntax_model.layer.W)

    def test_concat_op(self):
        a, b, c, d = Var('a'), Var('b'), Var('c'), Var('d')
        model = Concat(a, b) * Concat(c, d)
        aval, bval, cval, dval = [1.], [2.], [3.], [4.]
        y = model.forward_variables({'a': aval, 'b': bval, 'c': cval, 'd': dval})
        assert_array_equal(y, [3, 8])
        grads = model.backward_variables(1, 1)
        assert_array_equal(grads['a'], cval)
        assert_array_equal(grads['b'], dval)
        assert_array_equal(grads['c'], aval)
        assert_array_equal(grads['d'], bval)
