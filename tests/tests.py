# from __future__ import absolute_import

import inspect
import unittest

import numpy as np
import sys
from numpy.testing import assert_array_equal, assert_almost_equal, assert_array_almost_equal

from layers import Linear, Softmax, Sigmoid, Sign, Relu, Tanh, CheapTanh, Const, Mul, Sum, Wx
import numerical_gradient
from loss import NLL, CrossEntropyLoss, ClaudioMaxNLL, SquaredLoss
from optim import SGD
from network import Seq, Par, Map, Identity
from trainers import MinibatchTrainer, OnlineTrainer
from utils import make_one_hot_target


class LinearLayerTests(unittest.TestCase):
    def test_OneNeuronForward(self):
        layer = Linear(2, 1, initialize='ones')
        data = np.array([2., 2.])
        y = layer.forward(data)
        self.assertEqual(y, [5.0])

    def test_OneNeuronBackward(self):
        layer = Linear(2, 1, initialize='ones')
        x = np.array([2., 2.])
        y = layer.forward(x)
        self.assertEqual(y, [5.])

        dJdy = np.array([3])
        dxdy = layer.backward(dJdy)
        assert_array_equal(dxdy, [3., 3.])

    def test_OneNeuronUpdateGradient(self):
        layer = Linear(2, 1, initialize='ones')
        x = np.array([2., 2.])
        y = layer.forward(x)
        self.assertEqual(y, [5.])

        dJdy = np.array([3])
        dxdy = layer.backward(dJdy)
        assert_array_equal(dxdy, [3., 3.])

        update_grad = layer.calc_update_gradient(dJdy)
        assert_array_equal(layer.W + update_grad, np.array([[4, 7, 7]]))

    def test_TwoNeuronsForward(self):
        layer = Linear(2, 2, initialize='ones')
        data = np.array([.3, .3])
        y = layer.forward(data)
        assert_array_equal(y, [1.6, 1.6])

    def test_OneNeuronGradient(self):
        layer = Linear(2, 1)
        x = np.random.rand(2)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(1))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numerical_gradient.assert_are_similar(deriv_grad, numgrad[0])

    def test_TwoNeuronsGradient(self):
        layer = Linear(3, 2)
        x = np.random.rand(3)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(2))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numgrad = np.sum(numgrad, axis=0)
        numerical_gradient.assert_are_similar(deriv_grad, numgrad)

    def test_dtypes(self):
        x16 = np.array([1.99999999, 1.999, 1.9], dtype=np.float16)
        x64 = np.array([1.99999999, 1.999, 1.9], dtype=np.float64)

        l16 = Linear(3, 2, initialize='random', dtype=np.float16)
        self.assertEqual(l16.W.dtype, np.float16)
        y16 = l16.forward(x16)

        lNone = Linear(3, 2, initialize='ones')
        self.assertEqual(lNone.W.dtype, np.float64)
        yNone = lNone.forward(x64)

        l64 = Linear(3, 2, initialize='ones', dtype=np.float64)
        self.assertEqual(l64.W.dtype, np.float64)
        y64 = l64.forward(x64)

        assert_array_equal(yNone, y64)
        self.assertFalse(np.array_equal(y16, y64))

    def test_wx_numerical_grad(self):
        x = np.random.rand(3)
        wx = Wx(3, 5, initialize='ones')
        y = wx.forward(x)
        deriv_grad = wx.backward(np.ones(5))
        num_grad = numerical_gradient.calc(wx.forward, x)
        assert_array_almost_equal(deriv_grad, np.sum(num_grad, axis=0))


class SigmoidLayerTests(unittest.TestCase):
    def test_forward(self):
        layer = Sigmoid()
        x = np.array([2., 3., 4.])
        y = layer.forward(x)

    def test_backward(self):
        layer = Sigmoid()
        x = np.random.rand(2)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(1))

        numerical_grad_matrix = numerical_gradient.calc(layer.forward, x)

        # the numerical grad in this case is a matrix made of zeros with
        # dJ/dx_i only in the diagonal
        num_grad = np.diagonal(numerical_grad_matrix)

        numerical_gradient.assert_are_similar(deriv_grad, num_grad)


class TwoLinearLayersTests(unittest.TestCase):
    def test_Expand(self):
        model = Seq([
            Linear(2, 3, initialize='ones'),
            Linear(3, 1, initialize='ones')
        ])
        x = np.random.rand(2)
        model.forward(x)
        back = model.backward(np.array(1.))

    def test_Reduce(self):
        model = Seq([
            Linear(3, 2, initialize='ones'),
            Linear(2, 2, initialize='ones')
        ])
        x = np.random.rand(3)
        model.forward(x)
        model.backward(np.array([1., 1.]))

    def test_ManyErrors(self):
        model = Seq([
            Linear(2, 3, initialize='ones'),
            Linear(3, 1, initialize='ones')
        ])
        x = np.random.rand(2)
        y = model.forward(x)
        model.backward(np.array([1.]))


class SequentialModelTests(unittest.TestCase):
    def test_Linear(self):
        model = Seq()
        model.add(Linear(2, 1, initialize='ones'))
        data = np.array([2., 2.])
        y = model.forward(data)
        self.assertEqual(y, np.array([5]))

    def test_LinearSoftmax(self):
        model = Seq()
        model.add(Linear(2, 1))
        model.add(Softmax())
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(out, 1.)

    def test_LinearSigmoid(self):
        model = Seq()
        model.add(Linear(2, 1, initialize='ones'))
        model.add(Sigmoid())
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(round(out, 2), 1.)

    def test_short_syntax(self):
        model = Seq(Linear(2, 1, initialize='ones'), Sigmoid)
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(round(out, 2), 1.)

    def test_LinearLayerNumericalGradientCheck(self):
        x = np.random.rand(3)

        model = Seq()
        model.add(Linear(3, 2, initialize='ones'))

        num_grad = numerical_gradient.calc(model.forward, x)
        deriv_grad = model.backward(np.array([1, 1]))
        num_grad = np.sum(num_grad, axis=0)

        numerical_gradient.assert_are_similar(deriv_grad, num_grad)

    def test_TwoLinearSigmoidLayers(self):
        x = np.random.rand(5)

        real_model = Seq([
            Linear(5, 3, initialize='ones'),
            Sigmoid(),
            Linear(3, 5, initialize='ones'),
            Sigmoid()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5))

        num_model = Seq([
            Linear(5, 3, initialize='ones'),
            Sigmoid(),
            Linear(3, 5, initialize='ones'),
            Sigmoid()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)

        num_grad = np.sum(num_grad, axis=1)
        numerical_gradient.assert_are_similar(real_grad, num_grad)

    def test_TwoDifferentModelsShouldHaveDifferentGradients(self):
        x = np.random.rand(5)

        real_model = Seq([
            Linear(5, 3, initialize='ones'),
            Tanh(),
            Linear(3, 5, initialize='ones'),
            Tanh()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5))

        num_model = Seq([
            Linear(5, 3, initialize='ones'),
            Relu(),
            Linear(3, 5, initialize='ones'),
            Relu()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)
        num_grad = np.sum(num_grad, axis=1)
        self.assertFalse(numerical_gradient.are_similar(real_grad, num_grad))

    def test_TwoLinearLayersTanh(self):
        x = np.random.rand(5)

        real_model = Seq([
            Linear(5, 3, initialize='ones'),
            Tanh(),
            Linear(3, 5, initialize='ones'),
            Tanh()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5))

        num_model = Seq([
            Linear(5, 3, initialize='ones'),
            Tanh(),
            Linear(3, 5, initialize='ones'),
            Tanh()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)

        num_grad = np.sum(num_grad, axis=1)
        self.assertTrue(numerical_gradient.are_similar(real_grad, num_grad))


class ReluLayerTest(unittest.TestCase):
    def test_numerical_grad(self):
        layer = Relu()
        x = np.random.rand(5)
        layer.forward(x)
        grad = layer.backward(np.array([1.]))
        num_grad = numerical_gradient.calc(layer.forward, x)
        num_grad = num_grad.diagonal()
        numerical_gradient.assert_are_similar(grad, num_grad)


class CheapTanhTest(unittest.TestCase):
    def test_forward(self):
        x = np.array([-10, -0.5, 0, 0.5, 10])
        layer = CheapTanh()
        y = layer.forward(x)
        assert_array_equal(y, np.array([-1, -0.5, 0, 0.5, 1]))

    def test_numerical_grad(self):
        x = np.array([-100.34, -10, -0.5, 0, 0.5, 10, 130])

        for alpha in range(10):
            layer = CheapTanh(alpha)
            layer.forward(x)
            grad = layer.backward(np.ones(1))

            num_grad = numerical_gradient.calc(layer.forward, x).diagonal()
            assert_almost_equal(grad, num_grad)


class SoftmaxLayerTest(unittest.TestCase):
    def test_SoftmaxLayerGradientCheck(self):
        x = np.random.rand(3)
        layer = Softmax()
        layer.forward(x)
        grad = layer.backward(np.array([1.]))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numgrad = np.sum(numgrad, axis=1)
        numerical_gradient.assert_are_similar(grad, numgrad)


class NumericalGradient(unittest.TestCase):
    def test_calc_numerical_grad(self):
        f = lambda x: x ** 2
        x = np.array([1.5, 1.4])
        grad = np.array(2. * x)
        numgrad = numerical_gradient.calc(f, x)
        numgrad = np.diagonal(numgrad)
        numerical_gradient.assert_are_similar(grad, numgrad)

    def test_check_with_numerical_gradient(self):
        f = lambda x: x ** 2
        x = np.array([1.3, 1.4])
        grad = np.array(2. * x)
        numgrad = numerical_gradient.calc(f, x)
        numgrad = np.diagonal(numgrad)
        numerical_gradient.assert_are_similar(grad, numgrad)


class CrossEntropyLossTest(unittest.TestCase):
    def test_numerical_gradient(self):
        x = np.random.rand(5)
        target_class = make_one_hot_target(classes_n=5, target_class=1)

        loss = CrossEntropyLoss()
        y = loss.calc_loss(x, target_class)
        grad = loss.calc_gradient(y, target_class)

        def forward(i):
            return loss.calc_loss(i, target_class)

        num_grad = numerical_gradient.calc(forward, x)

        num_grad = np.sum(num_grad, axis=0)
        print num_grad
        numerical_gradient.assert_are_similar(grad, num_grad)


class NLLGradientTest(unittest.TestCase):
    def test_NLLNumericalGradient(self):
        nll = NLL()
        y = np.random.rand(3)
        t = int(2)
        nll.calc_loss(y, t)
        grad = nll.calc_gradient(y, t)

        def loss_with_target(x):
            return nll.calc_loss(x, t)

        num_grad = numerical_gradient.calc(loss_with_target, y).diagonal()
        assert_almost_equal(grad, num_grad, decimal=2)


class ClaudioMaxNLLGradientTest(unittest.TestCase):
    def test_ClaudioMaxNLLNumericalGradient(self):
        nll = ClaudioMaxNLL()
        y = np.random.rand(5)
        t = int(1)
        nll.calc_loss(y, t)
        grad = nll.calc_gradient(y, t)

        def loss_with_target(x):
            return nll.calc_loss(x, t)

        num_grad = numerical_gradient.calc(loss_with_target, y)
        num_grad = np.sum(num_grad, axis=0)
        numerical_gradient.assert_are_similar(grad, num_grad)


class MinibatchTrainerTest(unittest.TestCase):
    def test_CheckMinibatchTrainerEqualsSimpleTrainer(self):
        train_set = [(np.random.rand(2), i) for i in xrange(3)]
        loss = SquaredLoss()
        epochs = 1
        optimizer = SGD(learning_rate=0.01)

        minibatch_model = Seq([Linear(2, 5, initialize='ones')])
        minibatch_trainer = MinibatchTrainer()
        minibatch_trainer.train_minibatches(minibatch_model, train_set, batch_size=1,
                                            loss=loss, epochs=epochs, optimizer=optimizer, shuffle=False)

        simple_model = Seq([Linear(2, 5, initialize='ones')])
        simple_trainer = OnlineTrainer()
        simple_trainer.train(simple_model, train_set, loss, epochs, optimizer)

        x = np.random.rand(2)

        simple_y = simple_model.forward(x)
        minibatch_y = minibatch_model.forward(x)

        assert_array_equal(simple_y, minibatch_y)


class TestNumericGradientAllLayers(unittest.TestCase):
    def test_all(self):
        # Find all classes in layers.py
        all_layers = inspect.getmembers(sys.modules['layers'], inspect.isclass)
        excluded = ['Layer', 'Print', 'Store', 'Const', 'Linear', 'RegularizedLinear', 'Wx', 'Dropout', 'Sign',
                    'Softmax', 'ClaMax', 'Concat', 'Sum']

        x = np.random.rand(3)
        for class_name, layer_class in all_layers:
            if class_name in excluded:
                continue

            layer = layer_class()
            # print(class_name)

            y = layer.forward(x)
            grad = layer.backward(np.array(1))
            num_grad = numerical_gradient.calc(layer.forward, x)

            try:
                num_grad = num_grad.diagonal()
            except:
                pass
                # print('%s not diagonalized' % class_name)

            try:
                assert_almost_equal(grad, num_grad)
            except Exception as ex:
                print 'Exception in numerical gradient of %s Layer' % class_name
                raise ex


class SumLayerTest(unittest.TestCase):
    def test_forward(self):
        a = np.array([1.0, 1.0])
        b = np.array([2.1, 2.2])
        x = np.array([a, b])
        y = Sum().forward(x)
        assert_array_equal(y, np.array([3.1, 3.2]))


class ParallelTest(unittest.TestCase):
    def test_init_and_forward_SumLayer(self):
        # 1 + 2 + 3
        model = Seq(
            Par(Const(1.), Const(2.), Const(3.)),
            Sum
        )
        y = model.forward(np.array([1.]))
        assert_array_equal(y, 6.)

    def test_MulLayer(self):
        # 2 * 3 * 4
        model = Seq(
            Par(Const(2.), Const(3.), Const(4.)),
            Mul
        )
        y = model.forward(np.array([1.]))
        assert_array_equal(y, 24.)


class NetworkTests(unittest.TestCase):
    def test_map_forward(self):
        # [a, b] => [a, b]
        model = Map(Identity, Identity)
        a = np.array([2.])
        b = np.array([3.])
        y = model.forward(np.array([a, b]))
        assert_array_equal(y, np.array([[2.], [3.]]))

    def test_par_forward(self):
        # x => [x, x]
        model = Par(Identity, Identity)
        a = np.array([5, 1])
        b = np.array([6, 1])
        x = np.array([a, b])
        y = model.forward(x)
        assert_array_equal(y, np.array([x, x]))
        assert_array_equal(model.backward(1.), np.array([1, 1]))

    def test_map_sum_forward_backward(self):
        # [a, b] => a+b
        model = Seq(Map(Identity, Identity), Sum)
        a = np.array([2, 1])
        b = np.array([3, 1])
        y = model.forward(np.array([a, b]))
        assert_array_equal(y, np.array([5, 2]))
        grad = model.backward(np.array([1]))
        assert_almost_equal(grad, np.array([1, 1]))

    def test_map_mul_forward(self):
        # [a, b] => a*b
        model = Seq(Map(Identity, Identity), Mul)
        a = np.array([2., 1])
        b = np.array([3., 1])
        x = np.array([a, b])
        y = model.forward(x)
        assert_array_equal(y, np.array([6., 1.]))
        grad = model.backward(np.array([1.]))
        assert_array_equal(grad, np.array(b, a))

    def test_map_mul_backward(self):
        model = Seq(Map(Identity, Identity), Mul)
        a = np.array([2, 1])
        b = np.array([3, 1])
        x = np.array([a, b])
        y = model.forward(x)
        grad = model.backward(np.ones(1))
        # grad = np.array([b, a])
        num_grad = numerical_gradient.calc(model.forward, x)
        assert_almost_equal(grad, num_grad)


# class VanillaRNNTests(unittest.TestCase):
#     def test_forward(self):
#         model = VanillaRNN(3)
#         for i in range(3):
#             x = np.random.rand(3)
#             model.forward(x)
#             last_h = model.h_store.read()
#
#     def test_backward(self):
#         x = np.random.rand(3)
#
#         model = VanillaRNN(3)
#         model.forward(x)
#         dJdy = np.ones(3)
#         grad = model.backward(dJdy)
#
#         num_model = VanillaRNN(3)
#         num_grad = numerical_gradient.calc(num_model.forward, x)
#
#         assert_almost_equal(grad, num_grad)
