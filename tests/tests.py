import unittest

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_array_equal

from main import LinearLayer, SequentialModel, SoftmaxLayer, SigmoidLayer, SignLayer, SquaredLoss, ReluLayer, NLL, \
    PrintLayer, TanhLayer
import numerical_gradient


class LinearLayerTests(unittest.TestCase):

    def test_OneNeuronForward(self):
        layer = LinearLayer(2, 1, initialize='ones')
        data = np.array([2., 2.])
        y = layer.forward(data)
        print(y)
        self.assertEqual(y, [5.0])

    def test_OneNeuronBackward(self):
        layer = LinearLayer(2, 1, initialize='ones')
        x = np.array([2., 2.])
        y = layer.forward(x)
        self.assertEqual(y, [5.])

        dJdy = np.array([3])
        dxdy = layer.backward(dJdy)
        assert_array_equal(dxdy, [3., 3.])

    def test_OneNeuronUpdate(self):
        layer = LinearLayer(2, 1, initialize='ones')
        x = np.array([2., 2.])
        y = layer.forward(x)
        self.assertEqual(y, [5.])

        dJdy = np.array([3])
        dxdy = layer.backward(dJdy)
        assert_array_equal(dxdy, [3., 3.])

        layer.update(dJdy)
        assert_array_equal(layer.W, np.array([[4, 7, 7]]))

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
        self.assertTrue(numerical_gradient.are_similar(deriv_grad, numgrad[0]))

    def test_TwoNeuronsGradient(self):
        layer = LinearLayer(3, 2)
        x = np.random.rand(3)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(2))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numgrad = np.sum(numgrad, axis=0)
        self.assertTrue(numerical_gradient.are_similar(deriv_grad, numgrad))

class SigmoidLayerTests(unittest.TestCase):

    def test_forward(self):
        layer = SigmoidLayer()
        x = np.array([2., 3., 4.])
        y = layer.forward(x)
        print(y)

    def test_backward(self):
        layer = SigmoidLayer()
        x = np.random.rand(2)
        y = layer.forward(x)
        deriv_grad = layer.backward(np.ones(1))

        numerical_grad_matrix = numerical_gradient.calc(layer.forward, x)

        # the numerical grad in this case is a matrix made of zeros with
        # dJ/dx_i only in the diagonal
        numerical_grad = np.diagonal(numerical_grad_matrix)

        self.assertTrue(numerical_gradient.are_similar(deriv_grad, numerical_grad))


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
        print(y)
        model.backward(np.array([1.]))


class SequentialModelTests(unittest.TestCase):

    def test_wrong_input(self):
        model = SequentialModel([

        ])

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
        self.assertEqual(round(out,2), 1.)

    def test_trainLinearSigmoid(self):
        model = SequentialModel()
        model.add(LinearLayer(2, 1, initialize='ones'))
        model.add(SigmoidLayer())
        data = np.array([2., 3.])
        out = model.forward(data)
        self.assertEqual(round(out,2), 1.)

    def test_LinearLayerNumericalGradientCheck(self):
        x = np.random.rand(3)

        model = SequentialModel()
        model.add(LinearLayer(3, 2, initialize='ones'))

        num_grad = numerical_gradient.calc(model.forward, x)
        deriv_grad = model.backward(np.array([1, 1]))
        num_grad = np.sum(num_grad, axis=0)

        self.assertTrue(numerical_gradient.are_similar(deriv_grad, num_grad))

    def test_TwoLinearSigmoidLayers(self):
        x = np.random.rand(5)

        real_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            SigmoidLayer(),
            LinearLayer(3, 5, initialize='ones'),
            SigmoidLayer()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5))


        num_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            SigmoidLayer(),
            LinearLayer(3, 5, initialize='ones'),
            SigmoidLayer()
        ])
        num_grad = numerical_gradient.calc(num_model.forward, x)

        num_grad = np.sum(num_grad, axis=1)
        self.assertTrue(numerical_gradient.are_similar(real_grad, num_grad))

    def test_TwoDifferentModelsShouldHaveDifferentGradients(self):
        x = np.random.rand(5)

        real_model = SequentialModel([
            LinearLayer(5, 3, initialize='ones'),
            TanhLayer(),
            LinearLayer(3, 5, initialize='ones'),
            TanhLayer()
        ])
        y = real_model.forward(x)
        real_grad = real_model.backward(np.ones(5))

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
        real_grad = real_model.backward(np.ones(5))

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
        x = np.array([-13.3, -0.2, 53])
        layer.forward(x)
        grad = layer.backward(np.array([1.]))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numgrad = numgrad.diagonal()
        self.assertTrue(numerical_gradient.are_similar(grad, numgrad))


class SoftmaxLayerTest(unittest.TestCase):

    def test_SoftmaxLayerGradientCheck(self):
        x = np.random.rand(3)
        layer = SoftmaxLayer()
        layer.forward(x)
        grad = layer.backward(np.array([1.]))
        numgrad = numerical_gradient.calc(layer.forward, x)
        numgrad = np.sum(numgrad, axis=1)
        self.assertTrue(numerical_gradient.are_similar(grad, numgrad))


# class ClaMaxLayerTest(unittest.TestCase):
#
#     def test_ClaMaxGradientCheck(self):
#         x = np.random.rand(3)
#         layer = ClaMaxLayer()
#         layer.forward(x)
#         grad = layer.backward(np.array([1.]))
#         numgrad = numerical_gradient.calc(layer.forward, x)
#         self.assertTrue(numerical_gradient.are_similar(grad, numgrad))


class NumericalGradient(unittest.TestCase):

    def test_calc_numerical_grad(self):
        f = lambda x: x**2
        x = np.array([1.5, 1.4])
        grad = np.array(2.*x)
        numgrad = numerical_gradient.calc(f, x)
        numgrad = np.diagonal(numgrad)
        self.assertTrue(numerical_gradient.are_similar(grad, numgrad))

    def test_check_with_numerical_gradient(self):
        f = lambda x: x**2
        x = np.array([1.3, 1.4])
        grad = np.array(2.*x)
        numgrad = numerical_gradient.calc(f, x)
        numgrad = np.diagonal(numgrad)
        self.assertTrue(numerical_gradient.are_similar(grad, numgrad))


class NLLGradientTest(unittest.TestCase):

    def test_NLLNumericalGradient(self):
        nll = NLL()
        y = np.random.rand(3)
        t = int(2)
        nll.calc_loss(y, t)
        grad = nll.calc_gradient(y, t)

        def loss_with_target(x):
            return nll.calc_loss(x, t)

        num_grad = numerical_gradient.calc(loss_with_target, y)
        self.assertTrue(numerical_gradient.are_similar(grad, num_grad))

