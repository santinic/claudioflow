import unittest

from utils import sliding_window
from numpy.testing import assert_array_equal, assert_array_almost_equal


class SlidingWindowTest(unittest.TestCase):
    def test_one(self):
        print()

    def test_sliding_window(self):
        l = range(7)
        self.assertEqual(sliding_window(l, 3, 1), [0, 1, 2])
        self.assertEqual(sliding_window(l, 3, 1), [1, 2, 3])

# class SplitTest(unittest.TestCase):
#     def test_split(self):
#         assert_array_equal(split(range(10), [2, 3, 2, 2, 1]), [[0, 1], [2, 3, 4], [5, 6], [7, 8], [9]])
# assert_array_almost_equal(split(np.array(range(10)), [2, 3, 2, 2, 1]), [[0, 1], [2, 3, 4], [5, 6], [7, 8], [9]])
