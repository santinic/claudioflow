import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from layers import LinearLayer, SoftmaxLayer, SigmoidLayer, SignLayer, ReluLayer, TanhLayer
import numerical_gradient
from loss import NLL, CrossEntropyLoss, ClaudioMaxNLL
from sequential import SequentialModel


class LinearLayerTests(unittest.TestCase):
    def test_OneNeuronForward(self):
        layer = LinearLayer(2, 1, initialize='ones')
        data = np.array([2., 2.])
        y = layer.forward(data)
        self.assertEqual(y, [5.0])

    def test_OneNeuronBackward(self):
        layer = LinearLayer(2, 1, initialize='ones')
        x = np.array([2., 2.])
        y = layer.forward(x)
        self.assertEqual(y, [5.])

        dJdy = np.array([3])
        dxdy = layer.backward(dJdy)
        assert_array_equal(dxdy, [3., 3.])

    def test_OneNeuronUpdateGradient(self):
        layer = LinearLayer(2, 1, initialize='ones')
        x = np.array([2., 2.])
        y = layer.forward(x)
        self.assertEqual(y, [5.])

        dJdy = np.array([3])
        dxdy = layer.backward(dJdy)
        assert_array_equal(dxdy, [3., 3.])

        update_grad = layer.update_gradient(dJdy)
        assert_array_equal(layer.W + update_grad, np.array([[4, 7, 7]]))

    def test_TwoNeuronsForward(self):
        layer = LinearLayer(2, 2, initialize='ones')
        data = np.array([.3, .3])
        y = layer.forward(data)
        assert_array_equal(y, [1.6, 1.6])

    def test_OneNeuronGradient(self):
        layer = LinearLayer(2, 1)
        x = np.random.rand(2)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(1))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numerical_gradient.assert_are_similar(deriv_grad, numgrad[0])

    def test_TwoNeuronsGradient(self):
        layer = LinearLayer(3, 2)
        x = np.random.rand(3)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(2))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numgrad = np.sum(numgrad, axis=0)
        numerical_gradient.assert_are_similar(deriv_grad, numgrad)


class SigmoidLayerTests(unittest.TestCase):
    def test_forward(self):
        layer = SigmoidLayer()
        x = np.array([2., 3., 4.])
        y = layer.forward(x)

    def test_backward(self):
        layer = SigmoidLayer()
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
        model = SequentialModel([
            LinearLayer(2, 3, initialize='ones'),
            LinearLayer(3, 1, initialize='ones')
        ])
        x = np.random.rand(2)
        model.forward(x)
        back = model.backward(np.array(1.))

    def test_Reduce(self):
        model = SequentialModel([
            LinearLayer(3, 2, initialize='ones'),
            LinearLayer(2, 2, initialize='ones')
        ])
        x = np.random.rand(3)
        model.forward(x)
        model.backward(np.array([1., 1.]))

    def test_ManyErrors(self):
        model = SequentialModel([
            LinearLayer(2, 3, initialize='ones'),
            LinearLayer(3, 1, initialize='ones')
        ])
        x = np.random.rand(2)
        y = model.forward(x)
        model.backward(np.array([1.]))


class SequentialModelTests(unittest.TestCase):
    def test_Linear(self):
        model = SequentialModel()
        model.add(LinearLayer(2, 1, initialize='ones'))
        data = np.array([2., 2.])
        y = model.forward(data)
        self.assertEqual(y, np.array([5]))

    def test_LinearSoftmax(self):
        model = SequentialModel()
        model.add(LinearLayer(2, 1))
        model.add(SoftmaxLayer())
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(out, 1.)

    def test_LinearSigmoid(self):
        model = SequentialModel()
        model.add(LinearLayer(2, 1, initialize='ones'))
        model.add(SigmoidLayer())
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(round(out, 2), 1.)

    def test_trainLinearSigmoid(self):
        model = SequentialModel()
        model.add(LinearLayer(2, 1, initialize='ones'))
        model.add(SigmoidLayer())
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(round(out, 2), 1.)

    def test_LinearLayerNumericalGradientCheck(self):
        x = np.random.rand(3)

        model = SequentialModel()
        model.add(LinearLayer(3, 2, initialize='ones'))

        num_grad = numerical_gradient.calc(model.forward, x)
        deriv_grad = model.backward(np.array([1, 1]), backward_first_layer=True)
        num_grad = np.sum(num_grad, axis=0)

        numerical_gradient.assert_are_similar(deriv_grad, num_grad)

    def test_TwoLinearSigmoidLayers(self):
        x = np.random.rand(5)

        real_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            SigmoidLayer(),
            LinearLayer(3, 5, initialize='ones'),
            SigmoidLayer()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5), backward_first_layer=True)

        num_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            SigmoidLayer(),
            LinearLayer(3, 5, initialize='ones'),
            SigmoidLayer()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)

        num_grad = np.sum(num_grad, axis=1)
        numerical_gradient.assert_are_similar(real_grad, num_grad)

    def test_TwoDifferentModelsShouldHaveDifferentGradients(self):
        x = np.random.rand(5)

        real_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            TanhLayer(),
            LinearLayer(3, 5, initialize='ones'),
            TanhLayer()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5), backward_first_layer=True)

        num_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            ReluLayer(),
            LinearLayer(3, 5, initialize='ones'),
            ReluLayer()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)
        num_grad = np.sum(num_grad, axis=1)
        self.assertFalse(numerical_gradient.are_similar(real_grad, num_grad))

    def test_TwoLinearLayersTanh(self):
        x = np.random.rand(5)

        real_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            TanhLayer(),
            LinearLayer(3, 5, initialize='ones'),
            TanhLayer()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5), backward_first_layer=True)

        num_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            TanhLayer(),
            LinearLayer(3, 5, initialize='ones'),
            TanhLayer()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)

        num_grad = np.sum(num_grad, axis=1)
        self.assertTrue(numerical_gradient.are_similar(real_grad, num_grad))


class ReluLayerTest(unittest.TestCase):
    def test_ReluNumericalGradient(self):
        layer = ReluLayer()
        x = np.random.rand(5)
        layer.forward(x)
        grad = layer.backward(np.array([1.]))
        num_grad = numerical_gradient.calc(layer.forward, x)
        num_grad = num_grad.diagonal()
        numerical_gradient.assert_are_similar(grad, num_grad)


class SoftmaxLayerTest(unittest.TestCase):
    def test_SoftmaxLayerGradientCheck(self):
        x = np.random.rand(3)
        layer = SoftmaxLayer()
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
        target_class = 1

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
        # num_grad = np.sum(num_grad, axis=0)
        numerical_gradient.assert_are_similar(grad, num_grad)
