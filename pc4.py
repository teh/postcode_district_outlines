import time
import logging
import itertools
import collections
import csv

import pylab
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

    '\x00\x01\x01\x00': [(-1, 0), (1, 0),  (0, -1), (0, 1)],
    '\x01\x00\x00\x01': [(-1, 0), (1, 0),  (0, -1), (0, 1)],
    '\x01\x01\x01\x01': [],
}

def get_labeled_data_from_csv(infile):
    label_generator = itertools.count()
    label_map = collections.defaultdict(label_generator.next)

    # The csv headers we care about are:
    # Postcode, Positional_quality_indicator, Eastings, Northings, [...]
    return np.array([
        (label_map[row[0][:4].strip()], int(row[2]), int(row[3]))
        for row in csv.reader(infile)
        if int(row[2]) != 0 # some invalid rows in there
    ]), label_map

#import profilehooks
#@profilehooks.profile
def get_boundaries(data, classifier, step, x0, y0):
    # 1) Initialisation: find a boundary
    i, j = 0, 0

    while True:
        x, y = x0 + step * i, y0 + step * j
        prediction = classifier.predict(
            np.array([[x, y], [x+1, y], [x, y+1], [x+1,y+1]])
        )
        if len(set(prediction)) > 1:
            break
        i += 1

    work = set([(i, j)])
    seen = work.copy()

    # [63215  8195] [ 655448 1213660]
    xs, ys = data[:,1:].min(0)
    xe, ye = data[:,1:].max(0)

    minx, maxx, miny, maxy = x0, x0, y0, y0
    while work:
        i, j = work.pop()
        seen.add((i, j))
        x, y = x0 + step * i, y0 + step * j
        if x < xs or x > xe or y < ys or y > ye:
            continue

        prediction = classifier.predict(
            np.array([[x, y], [x+step, y], [x, y+step], [x+step,y+step]])
        )
        #print "P", prediction
        labels = set(prediction)
        for label in labels:
            lookup = (prediction == label).astype('u1').tostring()
            #print label, repr(lookup)
            for rel_i, rel_j in MARCHING_SQUARE_LOOKUP[lookup]:
                #print rel_i, rel_j
                new = (i + rel_i, j + rel_j)
                if new not in seen:
                    work.add(new)

    s = np.array(list(seen))
    print s.min(0), s.max(0), s.shape
    pylab.plot(s[:,0], s[:,1], 'b.')
    pylab.show()
    from IPython.Shell import IPShellEmbed;ipshell = IPShellEmbed([]);ipshell()

def main():
    #data, label_map = get_labeled_data_from_csv(open('all.csv'))
    #np.save('data.npy', data)
    data = np.load('data.npy')
    step = 50

    classifier = neighbors.KNeighborsClassifier(5, algorithm='kd_tree')
    classifier.fit(data[:,1:], data[:,0])
    x0, y0 = data[:,1:].mean(0).astype('u8')
    get_boundaries(data, classifier, step, x0, y0)

if __name__ == '__main__':
    logging.basicConfig(level=10)
    main()
