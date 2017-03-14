def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def partition(l, n):
    chunk_size = int(len(l) / n)
    return list(chunks(l, chunk_size))[:n]
