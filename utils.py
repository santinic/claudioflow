import unittest

import numpy as np


def chunks(l, chunk_size):
    """Yield successive chunks of size chunk_size from l"""
    for i in xrange(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def partition(l, n):
    """Divides the list l in n chunks"""
    chunk_size = int(len(l) / n)
    return list(chunks(l, chunk_size))[:n]


def slice_percentage(l, pc):
    """Returns the first pc percentage of the list l"""
    units = int((float(len(l)) / 100.) * (pc * 100))
    return l[:units]


def split_array_into_variable_sizes(a, sizes):
    '''Split array a in arrays of variable sizes'''
    ret = []
    for size in sizes:
        head, a = a[:size], a[size:]
        ret.append(np.array(head))
    return ret


def make_one_hot_target(classes_n, target_class):
    target_class_int = int(target_class)
    one_hot = np.zeros(classes_n)
    one_hot[target_class_int] = 1
    return one_hot


def to_one_hot_vector_targets(classes_n, train_set):
    return [(x, make_one_hot_target(classes_n, t)) for x, t in train_set]


def sliding_window(seq, window_size, step_size=1):
    seq_len = len(seq)
    while True:
        yield seq[p:p + window_size], seq[p + step_size:p + window_size + step_size]
        p = (p + 1) % seq_len

