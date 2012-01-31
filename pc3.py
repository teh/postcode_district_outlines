import time
import logging
import itertools
import collections
import csv

import numpy as np
from sklearn import neighbors
import networkx
import shapely.ops

# See http://en.wikipedia.org/wiki/Marching_squares
# I'm using a different mapping though.

MARCHING_SQUARE_LOOKUP = {
    '\x00\x00\x00\x00': [],

    '\x00\x00\x00\x01': [(0.5, 1.0), (1.0, 0.5)],
    '\x00\x00\x01\x00': [(0.0, 0.5), (0.5, 1.0)],
    '\x00\x01\x00\x00': [(0.5, 0.0), (1.0, 0.5)],
    '\x01\x00\x00\x00': [(0.0, 0.5), (0.5, 0.0)],

    '\x00\x00\x01\x01': [(0.0, 0.5), (1.0, 0.5)],
    '\x01\x01\x00\x00': [(0.0, 0.5), (1.0, 0.5)],
    '\x00\x01\x00\x01': [(0.5, 0.0), (0.5, 1.0)],
    '\x01\x00\x01\x00': [(0.5, 0.0), (0.5, 1.0)],

    '\x00\x01\x01\x01': [(0.0, 0.5), (0.5, 0.0)],
    '\x01\x00\x01\x01': [(0.5, 0.0), (1.0, 0.5)],
    '\x01\x01\x00\x01': [(0.0, 0.5), (0.5, 1.0)],
    '\x01\x01\x01\x00': [(0.5, 1.0), (1.0, 0.5)],

    '\x00\x01\x01\x00': [(0.0, 0.5), (0.5, 0.0),  (0.5, 1.0), (1.0, 0.5)],
    '\x01\x00\x00\x01': [(0.0, 0.5), (0.5, 1.0),  (0.5, 0.0), (1.0, 0.5)],
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
    
def get_boundary_for_label(data, classifier, num_label, step):
    # See
    # http://en.wikipedia.org/wiki/Postcodes_in_the_United_Kingdom#Operation_and_application
    # for the various divisions.
    t_start = time.time()
    district = data[data[:,0] == num_label, 1:]

    # Align grid to nearest "step". Also grow border by 25 units to
    # to make sure the marching squares can build a full loop.
    x0, y0 = np.floor(district.min(0) / step - 25) * step
    x1, y1 = np.ceil(district.max(0) / step + 25) * step

    # Use KNN to colour a grid that covers the district
    xx, yy = np.mgrid[x0:x1:step, y0:y1:step]
    prediction = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Split predicted labels into inside/outside
    prediction = (prediction == num_label).astype('u1')

    # We transpose to make reasoning about the lookups easier.
    prediction = prediction.transpose()

    # zero-pad predictions to make sure marching squares creates
    # closed outlines.
    tmp = np.zeros((prediction.shape[0] + 2, prediction.shape[1] + 2), dtype='u1')
    tmp[1:-1,1:-1] = prediction
    prediction = tmp

    outline = networkx.Graph()

    h, w = prediction.shape
    
    for i, j in np.ndindex(h - 1, w - 1):
        # We use tostring() as a cheap, hashable lookup type for the
        # marching squared implementation.

        # Dimension 0 ~ y ~ i, dim 1 ~ x ~ j:
        piter = iter(MARCHING_SQUARE_LOOKUP[prediction[i:i+2,j:j+2].tostring()])

        for rel1, rel2 in zip(piter, piter):
            p1 = int(x0 + step * (j + rel1[0])), int(y0 + step * (i + rel1[1]))
            p2 = int(x0 + step * (j + rel2[0])), int(y0 + step * (i + rel2[1]))

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
    except:
        logging.exception('ignoring.')

from multiprocessing import Pool

def all_boundaries(data, label_map):    
    step = 10
    classifier = neighbors.KNeighborsClassifier(5)
    classifier.fit(data[:,1:], data[:,0])

    #d  = [(x, label_map[x]) for x in ['N4', 'N16', 'N15', 'E5']]
    
    pool = Pool(2)
    results = [
        pool.apply_async(one_boundary, args=(data, classifier, num_label, text_label, step))
        for text_label, num_label in label_map.items()
    ]
    for r in results:
        print r.get()

def main():
    data, label_map = get_labeled_data_from_csv(open('all.csv'))
    all_boundaries(data, label_map)

if __name__ == '__main__':
    logging.basicConfig(level=10)
    main()