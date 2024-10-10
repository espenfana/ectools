'''testing ectools on data in the data folder'''

import os
import random

# Check if running in a Codespace
if 'CODESPACE_NAME' in os.environ:
    os.chdir('/workspaces')
print(f"Current working directory: {os.getcwd()}")

import ectools as ec
from ectools.helper_functions import filename_parser

# Set the plotter configuration
ec.set_config('plotter', ec.Plotter.BOKEH)

# Define the file path and name relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
FPATH = os.path.normpath(os.path.join(script_dir, 'data/'))
FNAME = '241007_01_MCL21_cswWE1_CV2_CO2_750C.DTA'

# Print the full path to the file for debugging
full_path = os.path.join(FPATH, FNAME)
print(f"Full path to the file: {full_path}")

# Importing and plotting single file
try:
    imp = ec.EcImporter(fname_parser=filename_parser)
    f = imp.load_file(FPATH, FNAME)
    f.plot()
except Exception as e:
    print('Single file import failed')
    raise e

# Testing "file" methods
try:
    imp = ec.EcImporter(fname_parser=filename_parser)
    f = imp.load_file(FPATH, FNAME)
    f.set_area(2)
    r = random.randint(0, len(f.curr))
    assert f.curr[r] == 2 * f.curr_dens[r], f'Current density not correct, curr:{f.curr[r]}, curr_dens:{f.curr_dens[r]}'
except Exception as e:
    print('Methods testing failed')
    raise e

# f.plot(y='curr_dens', hue='cycle', cycles=[1, 2])

# Loading a folder and returning an EcList
try:
    fl = imp.load_folder(FPATH)
except Exception as e:
    print('Folder loading failed')
    raise e
