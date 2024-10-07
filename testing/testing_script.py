'''testing ectools on data in the data folder'''


import sys
import os
import random
os.chdir('/workspaces')
print(os.getcwd())

import ectools as ec
from ectools.helper_functions import filename_parser

# Add the parent directory to sys.path to allow relational import of ectools
sys.path.append(os.path.join(os.getcwd(), '..'))

# Now you can import the ectools package

ec.set_config('plotter', ec.Plotter.BOKEH)

FPATH = 'ectools/testing/data/'
FNAME = '240912_01_MCL16_CSWWE1_LSV100-INIT_CO2_750C.DTA'

# Importing and plotting single file
try:
    imp = ec.EcImporter(fname_parser=filename_parser)
    f = imp.load_file(FPATH, FNAME)
    f.plot()
except Exception as e:
    print('Single file import failed')
    raise e

# Testing methods
try:
    imp = ec.EcImporter(fname_parser=filename_parser)
    f = imp.load_file(FPATH, FNAME)
    f.set_area(2)
    r = random.randint(0, len(f.curr))
    assert f.curr[r] == 2 * f.curr_dens[r], f'Current density not correct, curr:{f.curr[r]}, curr_dens:{f.curr_dens[r]}'
except Exception as e:
    print('Methods testing failed')
    raise e

#f.plot(y='curr_dens',hue='cycle', cycles=[1,2])

# Loading a folder and returning an EcList
try:
    fl = imp.load_folder(FPATH)
except Exception as e:
    print('Folder loading failed')
    raise e
