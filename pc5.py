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
import shapely.geometry

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
def get_boundary_for_label(data, classifier, num_label, step):
    t_start = time.time()
    # 1) Initialisation: find a boundary
    i, j = 0, 0
    district = data[data[:,0] == num_label, 1:]

    # xx align cooreds
    x0, y0 = district.mean(0).astype('u8')

    # [63215  8195] [ 655448 1213660]
    xs, ys = data[:,1:].min(0)
    xe, ye = data[:,1:].max(0)

    while True:
        x, y = x0 + step * i, y0 + step * j
        prediction = classifier.predict(
            np.array([[x, y], [x+step, y], [x, y+step], [x+step,y+step]])
        )
        if len(set(prediction)) > 1:
            break
        if x < xs or x > xe or y < ys or y > ye:
            return []
        i += 1

    work = set([(i, j)])
    initial = (i, j)
    seen = set()

    minx, maxx, miny, maxy = x0, x0, y0, y0
    outline = networkx.Graph()
    while work:
        i, j = work.pop()
        # Make sure we add the final edge in the loop.
        if (i, j) != initial:
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

        piter = iter(MARCHING_SQUARE_LOOKUP[lookup])
        for rel1, rel2 in zip(piter, piter):
            p1 = x + rel1[0] * step / 2 , y + rel1[1] * step / 2
            p2 = x + rel2[0] * step / 2 , y + rel2[1] * step / 2
            outline.add_node(p1)
            outline.add_node(p2)
            outline.add_edge(p1, p2)

    # Pick the largest subgraph, other graphs are most likely outliers.
    logging.info(
        "%s: Found %s connected graphs in %.2fs",
        num_label,
        len(networkx.connected_component_subgraphs(outline)),
        time.time() - t_start,
    )
    largest = max(
        networkx.connected_component_subgraphs(outline),
        key=lambda x: x.size()
    )
    return list(shapely.ops.polygonize(largest.edges()))[0]

def one_boundary(data, classifier, num_label, text_label, step):
    logging.debug("Preparing %s, %s", text_label, num_label)
    try:
        poly = get_boundary_for_label(data, classifier, num_label, step)
        x, y = poly.boundary.xy
        coords = np.vstack([x, y]).transpose()
        np.save('out/{}.npy'.format(text_label), coords)
    except Exception:
        logging.exception('ignoring.')

def main():
    #data, label_map = get_labeled_data_from_csv(open('all.csv'))
    #np.save('data.npy', data)
    #pickle.dump(label_map, open('label_map.pickle', 'w'))

    data = np.load('data.npy')
    label_map = pickle.load(open('label_map.pickle'))
    step = 10

    classifier = neighbors.KNeighborsClassifier(5, algorithm='kd_tree')
    classifier.fit(data[:,1:], data[:,0])

    for text_label, num_label in label_map.items():
        one_boundary(data, classifier, num_label, text_label, step)

if __name__ == '__main__':
    logging.basicConfig(level=10)
    main()
