import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import re

import ectools as ec

fpath = 'data/'
fname = '240902_18_MCL13_cswWE_CV50_750C_CO2.DTA'

imp = ec.ecImporter()
f = imp.load_file(fpath, fname)

f.plot(hue='cycle')