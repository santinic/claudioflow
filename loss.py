import numpy as np


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

    def calc_loss(self, x, target_class):
        c = np.max(x)
        exp_x = np.exp(x - c)
        self.y = np.divide(exp_x, np.sum(exp_x))

        self.one_hot_target = self.make_one_hot_target(x, target_class)
        J = - self.one_hot_target * np.log(self.y)
        return J

    def calc_gradient(self, y, target_class):
        return self.y - self.one_hot_target

    # def calc_loss_gas(self, y, target_class):
    #     self.one_hot_target = self.make_one_hot_target(y, target_class)
    #     totlog = np.log(np.sum(np.exp(y)))
    #     return self.one_hot_target * (totlog - y)
    #
    # def calc_gradient_gas(self, y, target_class):
    #     exp_y = np.exp(y - np.max(y))
    #     self.y = exp_y / np.sum(exp_y)
    #     return self.y - self.one_hot_target

    @staticmethod
    def make_one_hot_target(x, target_class):
        target_class_int = int(target_class)
        one_hot = np.zeros(x.size)
        one_hot[target_class_int] = 1
        return one_hot


NLL = NegLogLikelihoodLoss


class ClaudioMaxNLL:
    def calc_loss(self, y, target_class):
        one_hot_target = CrossEntropyLoss.make_one_hot_target(y, target_class)
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
