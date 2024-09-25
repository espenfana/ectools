'''testing ectools on data in the data folder'''

#import os
#import re

#import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt

import ectools as ec

FPATH = 'data/'
FNAME = '240912_01_MCL16_CSWWE1_LSV100-INIT_CO2_750C.DTA'

imp = ec.EcImporter()
f = imp.load_file(FPATH, FNAME)

f.plot(y='curr_dens')

f.set_area(0.489)
#f.plot(y='curr_dens',hue='cycle', cycles=[1,2])
