import unittest

from itertools import islice

from utils import sliding_window
from numpy.testing import assert_array_equal, assert_array_almost_equal

class SlidingWindowTest(unittest.TestCase):
    def test_sliding_window(self):
        l = range(4)
        # print(l)
        it = sliding_window(l, 2, 1)
        targets = [c[1] for c in list(islice(sliding_window(l, 2, 1), 3))]
        # print targets
        self.assertEqual(targets, [[1,2],[2,3],[1,2]])


# class SplitTest(unittest.TestCase):
#     def test_split(self):
#         assert_array_equal(split(range(10), [2, 3, 2, 2, 1]), [[0, 1], [2, 3, 4], [5, 6], [7, 8], [9]])
# assert_array_almost_equal(split(np.array(range(10)), [2, 3, 2, 2, 1]), [[0, 1], [2, 3, 4], [5, 6], [7, 8], [9]])
