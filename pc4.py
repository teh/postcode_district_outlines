import time
import logging
import itertools
import collections
import csv

import numpy as np
from sklearn import neighbors
import networkx
import shapely.ops

# See http://en.wikipedia.org/wiki/Marching_squares for the general
# idea. This is a "lazy" marching squares implementation though.

MARCHING_SQUARE_LOOKUP = {
    '\x00\x00\x00\x00': [],

    '\x00\x00\x00\x01': [(0, 1), (1, 0)],
    '\x00\x00\x01\x00': [(-1, 0), (0, 1)],
    '\x00\x01\x00\x00': [(0, -1), (1, 0)],
    '\x01\x00\x00\x00': [(-1, 0), (0, -1)],

    '\x00\x00\x01\x01': [(-1, 0), (1, 0)],
    '\x01\x01\x00\x00': [(-1, 0), (1, 0)],
    '\x00\x01\x00\x01': [(0, -1), (0, 1)],
    '\x01\x00\x01\x00': [(0, -1), (0, 1)],

    '\x00\x01\x01\x01': [(-1, 0), (0, -1)],
    '\x01\x00\x01\x01': [(0, -1), (1, 0)],
    '\x01\x01\x00\x01': [(-1, 0), (0, 1)],
    '\x01\x01\x01\x00': [(0, 1), (1, 0)],

#    '\x00\x01\x01\x00': [(0.0, 0.5), (0.5, 0.0),  (0.5, 1.0), (1.0, 0.5)], # ignore for this implementation
#    '\x01\x00\x00\x01': [(0.0, 0.5), (0.5, 1.0),  (0.5, 0.0), (1.0, 0.5)],
    '\x01\x01\x01\x01': [],
}
