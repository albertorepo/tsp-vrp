from itertools import groupby


def unique(l):
    """Remove duplicated elements on a list l"""
    l.sort()
    return list(k for k, _ in groupby(l))


