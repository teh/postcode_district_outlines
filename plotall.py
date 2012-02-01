import pickle
import random
import pylab
import glob
import numpy as np
import hashlib

from common import get_coastline

def gencolor():
    c = lambda: '{:02x}'.format(random.randint(0, 255))
    return '#' + c() + c() + c()

def col(label):
    return '#' + hashlib.md5(label).hexdigest()[:6]

data = np.load('data.npy')
label_map = pickle.load(open('label_map.pickle'))

x, y = get_coastline().boundary.xy
#pylab.plot(x, y, 'b-')

show_district_points = False

for path in glob.glob('out/*.npy'):
    d = np.load(path)
    c = col(path)
    pylab.fill(d[:,0], d[:,1], color=c)
    x, y = d.mean(0)
    pylab.text(x, y, path.split('/')[1].split('.')[0], color='#000000')

    if show_district_points:
        num_label = label_map['BN16']
        district = data[data[:,0] == num_label, 1:]
        pylab.plot(district[:,0], district[:,1], 'bo')

pylab.show()
