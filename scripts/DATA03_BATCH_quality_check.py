# general tools
import os
import sys
from glob import glob
import numpy as np
sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du


for lead in range(2, 24):
    print('v3, train, lead {}'.format(lead))
    names = glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_{}_temp/TRAIN*lead{}.npy".format('v3', lead))
    for name in names:
        try:
            np.load(name)
        except:
            print(name)
            
            
for lead in range(2, 24):
    print('v3, valid, lead {}'.format(lead))
    names = glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_{}_temp/VALID*lead{}.npy".format('v3', lead))
    for name in names:
        try:
            np.load(name)
        except:
            print(name)         


            