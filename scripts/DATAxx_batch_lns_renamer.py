# general tools
import os
import sys
from glob import glob
import numpy as np
sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import re
import subprocess

prefix_old = '{}_day{:03d}_{}_indx{}_indy{}_lead{}.npy'
prefix_new = '{}_day{:03d}_lead{}_indx{}_indy{}_{}.npy'

for lead in range(2, 24):
    print('lead = {}'.format(lead))
    
    if lead < 10:
        file_length = 48
    else:
        file_length = 49

    names = sorted(glob("{}/*lead{}.npy".format(path_batch_v3, lead)))


    for i in range(len(names)):
        name = names[i][-file_length:]

        train_flag = name[:5]
        pos_flag = name[13:24]

        nums = re.findall(r'\d+', name)

        day = int(nums[0])
        indx = nums[1]
        indy = nums[2]

        name_old = prefix_old.format(train_flag, day, pos_flag, indx, indy, lead)
        name_new = prefix_new.format(train_flag, day, lead, indx, indy, pos_flag)

        filename_old = path_batch_v3 + name_old
        filename_new = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/' + name_new

        cmd = 'ln -s {} {}'.format(filename_old, filename_new)
        #print(cmd)
        subprocess.call(cmd, shell=True)


