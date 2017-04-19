import gzip
import unittest

import numpy as np

import cPickle as pickle

import sklearn


def normalize(x):
    min = np.min(x)
    max = np.max(x)
    scaled = (x - np.mean(x)) / (max - min)
    return scaled


class NormalizeTest(unittest.TestCase):
    def test_tries(self):
        x = np.array([-100., 1., 140., 30., 0, -500000, 900000000, 30, 20, 10, 70, 80, 300])
        normalized = normalize(x)
        self.assertTrue((normalized > -1).all())
        self.assertTrue((normalized < 1).all())
        print normalized
