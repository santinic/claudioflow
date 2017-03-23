from loss import CrossEntropyLoss


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


def to_one_hot_vector_targets(classes_n, train_set):
    return [(x, CrossEntropyLoss.make_one_hot_target(classes_n, t)) for x, t in train_set]