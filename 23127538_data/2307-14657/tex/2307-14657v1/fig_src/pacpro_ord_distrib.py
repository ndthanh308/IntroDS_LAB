import os
import json
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from os.path import isdir, isfile, join, basename, dirname, realpath


data_file = join(join(dirname(realpath(__file__)), 'data'), 'pacpro_ord_distrib.csv')
assert isfile(data_file)
dst_dir = join(Path(realpath(__file__)).parent.parent, 'fig')
assert isdir(dst_dir)

import csv

x = list()
y = list()

with open(data_file, 'r') as fp:
    reader = csv.reader(fp)
    for i, l in enumerate(reader):
        if i == 0: continue
        print('line[{}] = {}'.format(i, l))
        x.append(int(l[0]))
        y.append(int(l[2]))

raise NotImplementedError()

#plt.savefig(join(dst_dir, "pacpro_ord_distrib.pdf"), bbox_inches='tight')
