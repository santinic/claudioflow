import numpy as np

from layers import Softmax
from utils import make_one_hot_target


class SquaredLoss:
    def calc_loss(self, y, target):
        J = 0.5 * (target - y) ** 2
        return J

    def calc_gradient(self, y, target):
        dJdy = (y - target)
        return dJdy


class NegLogLikelihoodLoss:
    def calc_loss(self, y, target):
        J = - target * np.log(y)
        return J

    def calc_gradient(self, y, target):
        return - target / y


class CrossEntropyLoss:
    """This is the combination of a final SoftmaxLayer and a Negative Log-Likelihood loss function."""

    def calc_loss(self, x, one_hot_target):
        c = np.max(x)
        exp_x = np.exp(x - c)
        self.y = np.divide(exp_x, np.sum(exp_x))

        J = - one_hot_target * np.log(self.y)
        return J

    def calc_gradient(self, y, one_hot_target):
        return self.y - one_hot_target

    @staticmethod
    def test_score(model, test_set):
        test_err = 0.
        for x, target in test_set:
            y = model.forward(x)
            y = Softmax().forward(y)
            # print(y, np.argmax(y), np.argmax(target))
            if np.argmax(y) != np.argmax(target):
                test_err += 1.
        test_score = (1.0 - test_err / float(len(test_set))) * 100.0
        return test_score


NLL = NegLogLikelihoodLoss


class ClaudioMaxNLL:
    def calc_loss(self, y, target_class):
        one_hot_target = make_one_hot_target(y.size, target_class)
        self.y = y
        self.s = np.sum(y)
        J = - one_hot_target * (y / self.s)
        return J

    def calc_gradient(self, last_y, target_class):
        s = self.s
        y = self.y
        z = y / s ** 2
        diag = - ((1 / s) - z)

        m = np.zeros((y.size, y.size))
        for i, row in enumerate(m):
            for j, cell in enumerate(row):
                if i != j:
                    m[i][j] = z[i]
                else:
                    m[i][j] = diag[i]

        target_class = int(target_class)
        return m[target_class]
