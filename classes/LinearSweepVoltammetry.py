import numpy as np
import re

from .ElectroChemistry import ElectroChemistry
# CyclicVoltammetry Class
class CyclicVoltammetry(ElectroChemistry):
    '''Cyclic voltammetry file container'''

    # Class variables and constants
    identifiers = {'Cyclic Voltammetry', 'CV'} # Strings in the raw files which indicate the technique
    get_columns = {**ElectroChemistry.get_columns,
        'oxred': (r'ox/red',),
        'cat': (r'cat', r'cathodic'),
        'cycle': (r'cycle number', r'cycle'),
        }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict naming scheme. The values should be list-like to support multiple different regex identifiers, which are used in a re.match.    
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict
    
    
    # Initialize

    def __init__(self, *args, **kwargs):
        '''Create a Cyclic Voltammetry container'''
        super().__init__(*args, **kwargs)
        

    # Class methods
    def parse_meta_mpt(self):
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        self.scanrate = float(self.widthsep['dE/dt'][0][0])
        self.units['scanrate'] = self.widthsep['dE/dt unit'][0]
        self.pot_init = float(self.widthsep['Ei'][0][0])
        self.units['pot_init'] = self.widthsep['Ei'][1]
        self.pot_end = float(self.widthsep['Ef'][0][0])
        self.units['pot_end'] = self.widthsep['Ef'][1]

    def parse_meta_gamry(self):
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        # tabsep gamry metadata is in meta_dict
        metamap = {'scanrate': 'SCANRATE', 'pot_init': 'VINIT', 'pot_end':'VFINAL'}
        for key, label in metamap.items():
            self[key] = float(self.meta_dict[label]['value'])
            self.units[key] = re.search(r'\((.*?)\)', self.meta_dict[label]['description']).group(1)

    def plot(self, 
    ax=None, 
    x='pot', 
    y='curr', 
    color='tab:blue', 
    clause=None, 
    hue=None, 
    ax_kws={}, 
    **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        ax = super().plot(ax=ax, x=x, y=y, clause=clause, hue=hue, ax_kws=ax_kws, **kwargs)
        if hue:
            ax.legend(title=hue)
        return ax


