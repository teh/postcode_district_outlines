import random
import pylab
import glob
import numpy as np

colors = set([
    '#aa0000',
    '#00aa00',
    '#0000aa',
    '#aa00aa',
    '#aaaa00',
    '#00aaaa',
])

def gencolor():
    c = lambda: '{:02x}'.format(random.randint(0, 255))
    return '#' + c() + c() + c()

for path in glob.glob('out/*.npy'):
    d = np.load(path)
    c = gencolor()
    pylab.plot(d[:,0], d[:,1], color=c)
    x, y = d.mean(0)
    pylab.text(x, y, path[-8:-4], color=c)

pylab.show()
