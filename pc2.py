import pickle
import csv
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.ckdtree import cKDTree
from sklearn import neighbors
import pylab
from matplotlib.colors import ListedColormap
from scipy.ndimage import measurements
from shapely import ops
import networkx

def cache():
    r = csv.reader(open('all.csv'))
    n4 = np.array([(int(row[3]), int(row[2])) for row in r if row[0].startswith('N4')])
    np.save('n4.npy', n4)

    r = csv.reader(open('all.csv'))
    other = np.array([(int(row[3]), int(row[2])) for row in r])
    np.save('other.npy', other)

def cache2():
    r = csv.reader(open('all.csv'))
    other = np.array([row[0] for row in r])
    labels = dict((l, i) for i, l in enumerate(set(x[:4].strip() for x in other)))
    labels_real = np.array([labels[x[:4].strip()] for x in other])
    np.save('map.npy', labels_real)

    rlabels = dict((b, a) for a, b in labels.items())
    pickle.dump(rlabels, open('labels.pickle', 'w'))

def main():
    n4  = np.load('n4.npy')
    other = np.load('other.npy')
    omap = np.load('map.npy')
    labels = pickle.load(open('labels.pickle'))

    clf = neighbors.KNeighborsClassifier(5, weights='distance')
    clf.fit(other, omap)

    x0, y0 = n4.min(0)
    x1, y1 = n4.max(0)

    xx, yy = np.mgrid[x0-200:x1+200:10, y0-200:y1+200:10]
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    np.save('predict.npy', z)

def main2():
    n4  = np.load('n4.npy')
    other = np.load('other.npy')
    omap = np.load('map.npy')
    labels = pickle.load(open('labels.pickle'))

    x0, y0 = n4.min(0)
    x1, y1 = n4.max(0)
    STEP = 10
    xx, yy = np.mgrid[x0-200:x1+200:STEP, y0-200:y1+200:STEP]
    
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#00aa00', '#aa00aa', '#aa0000', '#0000aa'])

    z = np.load('predict.npy')
    zu = dict((l, i) for i, l in enumerate(np.unique(z)))
    zm = np.array([zu[x] for x in z])
    zo = zm.reshape(xx.shape)

    # See http://en.wikipedia.org/wiki/Marching_squares for
    # meaning. Cases are different from wikipedia example though.
    LOOKUP = {
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

    zo = (zo == 1).astype('u1')
    g = networkx.Graph()
    x0 -= 200
    y0 -= 200

    for i in xrange(zo.shape[1]-1):
        for j in xrange(zo.shape[0]-1):
            #print repr(zo[i:i+2,j:j+2].tostring())
            ps = iter(LOOKUP[zo[j:j+2,i:i+2].tostring()])
            for a, b in zip(ps, ps):
                ap = x0 + STEP * (j + a[1]), y0 + STEP * (i + a[0])
                bp = x0 + STEP * (j + b[1]), y0 + STEP * (i + b[0])

                g.add_node(ap)
                g.add_node(bp)
                g.add_edge(ap, bp)

    
    cc = networkx.connected_component_subgraphs(g)[0]
    from IPython.Shell import IPShellEmbed;ipshell = IPShellEmbed([]);ipshell()
    poly = list(ops.polygonize(cc.edges()))[0].simplify(5)
    x, y = poly.boundary.xy
    
    #pylab.pcolormesh(xx, yy, zo, cmap=cmap)
    pylab.fill(x, y, '#aaaaaa')
    pylab.plot(n4[:,0], n4[:,1], 'ro')
    pylab.show()

            
#cache2()
#main()
main2()
#cache()
