'''testing ectools on data in the data folder'''
import ectools as ec
from ectools.helper_functions import filename_parser

import sys
import os

# Add the parent directory to sys.path to allow relational import of ectools
sys.path.append(os.path.join(os.getcwd(), '..'))

# Now you can import the ectools package


ec.set_config('plotter', ec.Plotter.BOKEH)

FPATH = 'data/'
FNAME = '240912_01_MCL16_CSWWE1_LSV100-INIT_CO2_750C.DTA'


imp = ec.EcImporter(fname_parser=filename_parser)
f = imp.load_file(FPATH, FNAME)

f.plot_bokeh()

f.set_area(0.489)
#f.plot(y='curr_dens',hue='cycle', cycles=[1,2])

fl = imp.load_folder(FPATH)
