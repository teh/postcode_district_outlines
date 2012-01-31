import time
import logging
import itertools
import collections
import csv
import pickle

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
    ]), dict(label_map)

#import profilehooks
#@profilehooks.profile
def get_boundaries(data, classifier, step, num_label):
    # 1) Initialisation: find a boundary
    i, j = 0, 0
    district = data[data[:,0] == num_label, 1:]

    # xx align cooreds
    x0, y0 = district.mean(0).astype('u8')

    while True:
        x, y = x0 + step * i, y0 + step * j
        prediction = classifier.predict(
            np.array([[x, y], [x+step, y], [x, y+step], [x+step,y+step]])
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
        if num_label not in set(prediction):
            continue
        
        lookup = (prediction == num_label).astype('u1').tostring()
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
    #pickle.dump(label_map, open('label_map.pickle', 'w'))

    data = np.load('data.npy')
    label_map = pickle.load(open('label_map.pickle'))
    step = 10

    classifier = neighbors.KNeighborsClassifier(5, algorithm='kd_tree')
    classifier.fit(data[:,1:], data[:,0])

    get_boundaries(data, classifier, step, label_map['N4'])

if __name__ == '__main__':
    logging.basicConfig(level=10)
    main()
