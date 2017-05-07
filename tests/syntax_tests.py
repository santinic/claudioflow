import unittest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

import layers
import syntax
from network import Seq
from optim import SGD
from syntax import Var, Concat, Linear, Store, Sigmoid, SyntaxOp, Const, Tanh


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

    def test_merge_dicts_should_not_be_sensible_to_pointers(self):
        a1 = np.array([1, 1.])
        dicts = [
            {'a': a1, 'b': a1},
            {'a': a1}
        ]
        merged = SyntaxOp.merge_backprop_dicts(dicts)
        assert_array_equal(merged['a'], np.array([2, 2]))
        assert_array_equal(merged['b'], np.array([1, 1]))

    def test_basic(self):
        a_var = Var('a')
        b_var = Var('b')
        a = np.random.rand(5)
        b = np.random.rand(5)
        input_dict = {'a': a, 'b': b}

        # a+b
        model = (a_var + b_var)
        y = model.forward_variables(input_dict)
        assert_array_equal(y, a + b)
        grad = model.backward_variables(np.ones(5))
        assert_array_equal(grad['a'], [1, 1, 1, 1, 1])
        assert_array_equal(grad['b'], [1, 1, 1, 1, 1])

        # a+a+b
        model = (a_var + b_var + a_var)
        y = model.forward_variables(input_dict)
        assert_array_almost_equal(y, a + a + b)
        grad = model.backward_variables(np.ones(5), debug=True)
        assert_array_equal(grad['a'], [2, 2, 2, 2, 2])
        assert_array_equal(grad['b'], [1, 1, 1, 1, 1])

        # a-b
        model = (a_var - b_var - b_var)
        y = model.forward_variables(input_dict)
        assert_array_equal(y, a - 2 * b)
        grad = model.backward_variables(np.ones(5))
        assert_array_equal(grad['a'], [1, 1, 1, 1, 1])
        assert_array_equal(grad['b'], [-2, -2, -2, -2, -2])

        # a*a*b
        model = a_var * b_var * a_var

        y = model.forward_variables(input_dict)
        assert_array_equal(y, a * b * a)

        grad = model.backward_variables(np.ones(5))
        assert_array_almost_equal(grad['a'], 2 * a * b, decimal=12)
        assert_array_almost_equal(grad['b'], a * a, decimal=12)

    # def test_const(self):
    #     a_var = Var('a')
    #     b_var = Var('b')
    #     a = np.array([2., 3.])
    #     b = np.array([3., 5.])
    #
    #     # Const(18)*a-b
    #     model = Const(18.) * a_var - b_var
    #     y = model.forward_variables({'a': a, 'b': b})
    #     assert_array_equal(y, 18 * a - b)
    #     grad = model.backward_variables(np.ones(2))
    #     assert_array_equal(grad['a'], 18)
    #     assert_array_equal(grad['b'], -1)

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
        assert_array_almost_equal(grad['a'], 2 * a * sigm(a ** 2) * (1 - sigm(a ** 2)) + 1)
        assert_array_almost_equal(grad['b'], 3 * b * b)

    def test_proper_equation_sum(self):
        var_a = Var('a')
        var_b = Var('b')
        a = np.array([2.3, 3., 3])
        b = np.array([3., 5., 4])
        input_dict = {'a': a, 'b': b}
        sigm = layers.Sigmoid().forward

        model = Sigmoid(var_a + var_a)

        y = model.forward_variables(input_dict)
        assert_array_equal(y, sigm(a + a))

        grad = model.backward_variables(np.ones(3))
        assert_array_almost_equal(grad['a'], 2 * sigm(a + a) * (1 - sigm(a + a)))

    # TODO:
    # def test_matrix_plus_bias(self):
    #     W_val = np.random.rand(5, 10)
    #     b_val = np.random.rand(5)
    #     x_val = np.random.rand(5)
    #
    #     W = layers.MatrixWeight(5, 10, initialize=W_val.copy())
    #     b = layers.VectorWeight(5, initialize=b_val.copy())
    #     x = Var('x')
    #     model = W * x + b
    #     model_y = model.forward_variables({'x': x_val})
    #
    #     y = np.dot(W_val * x_val + b_val)
    #     assert_array_equal(y, model_y)



    def test_many_vector_expressions(self):
        a = Var('a')
        b = Var('b')
        av = np.random.rand(3)
        bv = np.random.rand(3)
        sigm = layers.Sigmoid().forward
        dJdy = np.ones(3)

        models = [
            (a + b, av + bv, (1, 1)),
            (a * a + b, av ** 2 + bv, (2 * av, 1)),
            (a + b + a, av + bv + av, (2, 1)),

            (Sigmoid(a + b), sigm(av + bv),
             (sigm(av + bv) * (1 - sigm(av + bv)), sigm(av + bv) * (1 - sigm(av + bv)))),

            (Sigmoid(a + b + a), sigm(av + bv + av),
             (2 * sigm(av + bv + av) * (1 - sigm(av + bv + av)),
              sigm(av + bv + av) * (1 - sigm(av + bv + av))))
        ]

        for model, fwd_return, (a_grad, b_grad) in models:
            y = model.forward_variables({'a': av, 'b': bv})
            assert_array_equal(y, fwd_return)

            grad = model.backward_variables(dJdy, debug=True)
            assert_array_almost_equal(grad['a'], a_grad, err_msg="wrong gradient in model: %s" % model)
            assert_array_almost_equal(grad['b'], b_grad, err_msg="wrong gradient in model: %s" % model)

    def test_tanh(self):
        x = Var('x')
        b = Var('b')
        model = Tanh(x + b)
        x_val = np.random.rand(3)
        b_val = np.random.rand(3)
        y = model.forward_variables({'x': x_val, 'b': b_val})
        assert_array_equal(y, np.tanh(x_val + b_val))

        dJdy = np.ones(3)
        grad = model.backward_variables(dJdy, debug=True)

        manual_grad = (1. - np.tanh(x_val + b_val) ** 2) * dJdy
        assert_array_equal(grad['x'], manual_grad)
        assert_array_equal(grad['b'], manual_grad)

    def test_tanh_tanh(self):
        x = Var('x')
        b = Var('b')
        model = Tanh(Tanh(x + b))
        x_val = np.random.rand(3)
        b_val = np.random.rand(3)
        y = model.forward_variables({'x': x_val, 'b': b_val})
        assert_array_equal(y, np.tanh(np.tanh(x_val + b_val)))

        dJdy = np.ones(3)
        grad = model.backward_variables(dJdy)

        y1 = np.tanh(x_val + b_val)
        manual_grad = (1 - y1 ** 2) * (1 - y ** 2) * dJdy
        assert_array_equal(grad['x'], manual_grad)
        assert_array_equal(grad['b'], manual_grad)

    def test_compare_linear_syntax_and_linear_layer(self):
        x = np.random.rand(3)
        syntax_model = syntax.WxBiasLinear(3, 4, initialize_W='ones', initialize_b='ones', input=Var('x'))
        layer_model = layers.Linear(3, 4, initialize='ones')
        optimizer = SGD(0.1)

        # W = np.ones((4, 3))
        # b = np.ones(4)

        for i in range(5):
            syntax_y = syntax_model.forward_variables({'x': x})
            layer_y = layer_model.forward(x)
            assert_array_almost_equal(syntax_y, layer_y, decimal=12)

            dJdy = np.random.rand(4)
            syntax_grad = syntax_model.backward_variables(dJdy)
            layer_grad = layer_model.backward(dJdy)
            self.assertEqual(syntax_grad['x'].shape, layer_grad.shape, 'gradients should have the same vector shape')
            assert_array_almost_equal(syntax_grad['x'], layer_grad)

            # real_y = W.dot(x) + b
            # real_grad = W.T.dot(dJdy)
            # assert_array_equal(real_y, syntax_y)
            # assert_array_equal(syntax_grad['x'], real_grad)

            syntax_model.update_weights(optimizer)
            layer_model.update_weights(optimizer)



    # def test_linear_and_store(self):
    #     var_a = Var('a')
    #     var_b = Var('b')
    #     a = np.array([2.3, 3., 3])
    #     b = np.array([3., 5., 4])
    #
    #     stored_ab = Store(3, var_a * var_b)
    #     linear = Linear(3, 3, initialize='ones', input=stored_ab)
    #     model = var_a + linear + var_a * var_b
    #
    #     y = model.forward_variables({'a': a, 'b': b})
    #     assert_array_equal(y, np.sum(a * b) + 1 + a + a * b)
    #
    #     grad = model.backward_variables(np.ones(1))
    #     assert_array_equal(grad['a'], b + 1 + b)
    #     assert_array_equal(grad['b'], a + a)
    #
    #     assert_array_equal(stored_ab.read_forward(), a * b)

    def test_update_weights_layer_vs_syntax(self):
        x = np.array([1., 2., 3.])
        optimizer = SGD(0.1)

        W = np.random.rand(3, 3 + 1)

        linear_layer = layers.Linear(3, 3, initialize=W.copy())
        linear_layer_model = Seq(linear_layer, layers.Tanh)
        y = linear_layer_model.forward(x)
        back = linear_layer_model.backward(np.ones(3))

        var_x = Var('x')
        syntax_linear = Linear(3, 3, initialize=W.copy(), input=var_x)
        syntax_model = Tanh(syntax_linear)
        syntax_y = syntax_model.forward_variables({'x': x})
        syntax_back = syntax_model.backward_variables(np.ones(3))

        assert_array_equal(linear_layer.delta_W, syntax_linear.layer.delta_W)

        # update weights in both models
        linear_layer_model.update_weights(optimizer)
        syntax_model.update_weights(optimizer)

        assert_array_equal(y, syntax_y)
        assert_array_equal(back, syntax_back['x'])
        assert_array_equal(linear_layer.W, syntax_linear.layer.W)

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
