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
        target = int(target)
        J = - np.log(y[target])
        return J

    def calc_gradient(self, y, target):
        # dJdy = - y
        # dJdy[target] += 1.
        # return dJdy
        pass


class SoftmaxNLL:
    def calc_loss(self, x, target_class):
        c = np.max(x)
        exp_x = np.exp(x - c)
        self.y = np.divide(exp_x, np.sum(exp_x))

        self.one_hot_target = self.make_one_hot_target(x, target_class)
        J = - self.one_hot_target * np.log(self.y)
        return J

    # def calc_loss_gas(self, x, target_class):
    #     self.one_hot_target = self.make_one_hot_target(x, target_class)


    def calc_gradient(self, y, target_class):
        return self.y - self.one_hot_target

    # def calc_loss(self, x, target_class):
    #     c = np.max(x)
    #     exp_x = np.exp(x - c)
    #     self.y = exp_x / np.sum(exp_x)
    #
    #     target_index = int(target_class)
    #     J = - np.log(self.y[target_index])
    #     return J
    #
    # def calc_gradient(self, y, target_class):
    #     target_index = int(target_class)
    #     self.y[target_index] -= 1
    #     return self.y

    @staticmethod
    def output(y):
        return np.argmax(y)

    @staticmethod
    def make_one_hot_target(x, target_class):
        target_class_int = int(target_class)
        one_hot = np.zeros(x.size)
        one_hot[target_class_int] = 1
        return one_hot


NLL = NegLogLikelihoodLoss
