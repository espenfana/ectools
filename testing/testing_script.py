'''testing ectools on data in the data folder'''

#import os
#import re

#import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt

import ectools as ec

FPATH = 'data/'
FNAME = '240902_18_MCL13_cswWE_CV50_750C_CO2.DTA'

imp = ec.EcImporter()
f = imp.load_file(FPATH, FNAME)

f.plot(y='curr_dens',hue='cycle')

f.set_area(0.489)
f.plot(y='curr_dens',hue='cycle', cycles=[1,2])
