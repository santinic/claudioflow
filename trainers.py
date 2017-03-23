import random
import timeit
from itertools import izip

import numpy as np
import os

from utils import chunks


class SimpleTrainer:
    def train(self, model, train_set, loss, epochs, optimizer, show_progress=False, save_progress=False):
        """This is "online learning": we backpropagate after forwarding one input_data at the time."""
        epochs_losses = []
        for epoch in xrange(epochs):
            if show_progress: print("Epoch %d/%d" % (epoch + 1, epochs))
            epoch_loss_sum = 0.
            for x, target in train_set:
                J, dJdy = self.train_one(model, x, target, loss, optimizer, show_progress=show_progress)
                epoch_loss_sum += J
            epoch_mean_loss = epoch_loss_sum / float(epochs)
            epochs_losses.append(epoch_mean_loss)
        return epochs_losses

    def train_one(self, model, x, target, loss, optimizer, show_progress=False):
        y = model.forward(x, is_training=True)
        J = loss.calc_loss(y, target)
        dJdy = loss.calc_gradient(y, target)
        model.backward_and_update(dJdy, optimizer)
        return J, dJdy

    # def test(self, input_data, target_data, loss):
    #     for x, target in izip(input_data, target_data):
    #         y = self.forward(x)
    #         print (y)
    #         output = loss.output(y)
    #         print output, target


class PlottableTrainer:
    def __init__(self):
        pass

        # def plot_errors_history(self):
        #     plt.figure()
        #     plt.title('Errors History (J)')
        #     plt.plot(xrange(len(self.errors_history)), self.errors_history, color='red')
        #     plt.ylim([0, 2])
        #     return plt

        # def plot_loss_gradient_history(self):
        #     plt.figure()
        #     plt.title('Loss Gradient History (dJ/dy)')
        #     plt.plot(xrange(len(self.loss_gradient_history)), self.loss_gradient_history, color='orange')
        #     plt.ylim([0, 2])
        #     return plt


class MinibatchTrainer:
    def train_minibatches(self, model, train_set,
                          batch_size, loss, epochs, optimizer, shuffle=True,
                          show_progress=False, save_progress=False):

        # copy train_set locally to shuffle it
        data = list(train_set)
        epoch_mean_losses = []
        for epoch in xrange(epochs):
            if shuffle:
                random.shuffle(data)

            batch_mean_losses_sum = 0.
            for batch in chunks(data, batch_size):
                batch_mean_loss = self.train_one_minibatch(model, batch, loss, optimizer)
                batch_mean_losses_sum += batch_mean_loss

            epoch_mean_loss = batch_mean_losses_sum / float(batch_size)
            epoch_mean_losses.append(epoch_mean_loss)

            if show_progress:
                print("Epoch %d/%d: Epoch Mean Loss = %f" % (epoch + 1, epochs, batch_mean_loss))
        return epoch_mean_losses

    def train_one_minibatch(self, model, batch, loss, optimizer=None):
        batch_mean_loss, mean_delta, mean_x = self.forward_one_minibatch(model, batch, loss)
        model.forward(mean_x)
        model.backward_and_update(mean_delta, optimizer)
        return batch_mean_loss

    def forward_one_minibatch(self, model, batch, loss):
        loss_sum = 0.
        delta_sum = 0.
        x_sum = 0.
        for couple in batch:
            x, target = couple
            y = model.forward(x)
            J = loss.calc_loss(y, target)
            dJdy = loss.calc_gradient(y, target)
            loss_sum += J
            delta_sum += dJdy
            x_sum += x
        batches_n = float(len(batch))
        mean_x = x_sum /batches_n
        mean_delta = delta_sum / batches_n
        mean_losses = loss_sum / batches_n
        batch_mean_loss = np.mean(mean_losses)
        return batch_mean_loss, mean_delta, mean_x

    train = train_minibatches


class PatienceTrainer(MinibatchTrainer):
    """This multibatch trainer uses a validation set and a patience variable to decide when to stop.
    Inspired by this Theano tutorial: http://deeplearning.net/tutorial/mlp.html
    """

    def train(self, model,
              train_batches, valid_batch, test_batch,
              batch_size, loss, max_epochs, optimizer,
              test_score_function,
              patience=10000, patience_increase=2, improvement_threshold=0.995):
        """
        :param model: the SequentialModel instance to use
        :param train_set:
        :param valid_set:
        :param test_set:
        :param batch_size:
        :param loss: loss instance to use, for example: patience=CrossEntropyLoss()
        :param max_epochs:
        :param optimizer: optimizer instance to use, for example: optimizer=SGD(learning_rate=0.5)
        :param patience: look as this many examples regardless
        :param patience_increase: wait this much longer when a new best is found
        :param improvement_threshold: a relative improvement of this much is considered significant
        :return:
        """
        n_train_batches = float(len(train_batches)) // batch_size

        # go through this many minibatches before checking the network on the validation set;
        # in this case we check every epoch
        validation_frequency = min(n_train_batches, patience // 2)

        best_valid_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < max_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index, minibatch in enumerate(list(chunks(train_batches, batch_size))):
                iter_i = (epoch - 1) * n_train_batches + minibatch_index
                train_batch_mean_loss = self.train_one_minibatch(model, minibatch, loss, optimizer)

                if (iter_i + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    mean_valid_loss, valid_delta, mean_x = self.forward_one_minibatch(model, valid_batch, loss)

                    print('epoch %i, minibatch %i/%i, validation error %f, last train batch err %f (patience %d)' %
                          (epoch, minibatch_index + 1, n_train_batches, mean_valid_loss, train_batch_mean_loss, patience))

                    # if we got the best validation score until now
                    if mean_valid_loss < best_valid_loss:
                        # improve patience if loss improvement is good enough
                        if mean_valid_loss < best_valid_loss * improvement_threshold:
                            patience = max(patience, iter_i * patience_increase)

                        best_valid_loss = mean_valid_loss
                        best_iter = iter_i

                        # test it on the test set
                        test_score = test_score_function(model, test_batch)

                        print('\tepoch %i, minibatch %i/%i, test error of best model %f %%' %
                              (epoch, minibatch_index + 1, n_train_batches, test_score))

                if patience <= iter_i:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete. Best validation score of %f %% '
              'obtained at iteration %i, with test performance %f %%' %
              (best_valid_loss, best_iter + 1, test_score))
        print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
