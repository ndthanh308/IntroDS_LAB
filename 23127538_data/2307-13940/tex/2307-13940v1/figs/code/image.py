#!/usr/bin/env python3

from pylab import imshow, show, get_cmap, savefig
from numpy import random

myimg = random.random((5,5))

imshow(myimg, cmap=get_cmap("Purples"), interpolation='nearest')
# show()
savefig("test.svg")
