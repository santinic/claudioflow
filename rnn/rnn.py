import numpy as np


class RNN:
    def forward_window(self, window, loss, is_training=True):
        loss_sum = 0.
        dJdys = []
        ys = []
        for node, (x, target) in zip(self.nodes, window):
            y = node.forward(x, is_training)
            J = loss.calc_loss(y, target)
            dJdy = loss.calc_gradient(y, target)
            dJdys.append(dJdy)
            loss_sum += np.sum(J)
        mean_loss = loss_sum / float(len(window))
        return ys, dJdys, mean_loss

    def backward_window(self, dJdys):
        dhnext = np.zeros_like(self.hidden_size)
        for node, dJdy in reversed(zip(self.nodes, dJdys)):
            dhnext = node.backward_through_time(dJdy, dhnext)

    def learn_window(self, window, loss, optimizer):
        rnn = self
        ys, dJdys, mean_loss = rnn.forward_window(window, loss)
        rnn.backward_window(dJdys)
        rnn.update_weights(optimizer)
        return mean_loss

    def update_weights(self, optimizer):
        self.nodes[0].update_weights(optimizer)
