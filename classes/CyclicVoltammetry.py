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
        self.pot_upper = float(self.widthsep['E1'][0][0])
        self.units['pot_upper'] = self.widthsep['E1'][1]
        self.pot_lower = float(self.widthsep['E2'][0][0])
        self.units['pot_lower'] = self.widthsep['E2'][1]
        self.pot_end = float(self.widthsep['Ef'][0][0])
        self.units['pot_end'] = self.widthsep['Ef'][1]
        self.ncycles = int(self.widthsep['nc cycles'][0][0])

    def parse_meta_gamry(self):
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        # tabsep gamry metadata is in meta_dict
        metamap = {'scanrate': 'SCANRATE', 'pot_init': 'VINIT', 'pot_upper':'VLIMIT1', 'pot_lower':'VLIMIT2', 'pot_end':'VFINAL', 'ncycles': 'CYCLES'}
        for key, label in metamap.items():
            self[key] = float(self.meta_dict[label]['value'])
            self.units[key] = re.search(r'\((.*?)\)', self.meta_dict[label]['description']).group(1)
        self.ncycles = int(self.ncycles)
        if self.pot_upper < self.pot_lower:
            tmp = self.pot_lower
            self.pot_lower = self.pot_upper
            self.pot_upper = tmp

    def plot(self, 
    ax=None, 
    x='pot', 
    y='curr', 
    color='tab:blue', 
    cycles=None, 
    clause=None, 
    hue=None, 
    ax_kws={}, 
    **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        if hue is True: #default hue
            hue = 'cycle'
        if cycles:
            clause = np.where(np.logical_and(self['cycle']>=cycles[0],  self['cycle']<=cycles[1]))
        ax = super().plot(ax=ax, x=x, y=y, clause=clause, hue=hue, ax_kws=ax_kws, **kwargs)
        if hue:
            ax.legend(title=hue)
        return ax

    def filter_cycle(self, column: str, cycles: list | int | str) -> np.ndarray:
        '''
        Filter column based on a condition.
        
        column: Data column to filter
        cycles:
            int: Single cycle
            list: List of cycles (in order)
            str: Condition, e.g., '<10', '>5', '2:4', 'n:', ':n'
        '''
        if isinstance(cycles, int):
            # Single cycle filtering
            return self[column][self.cycle == cycles]
        
        if isinstance(cycles, list):
            # List of cycles filtering
            return self[column][np.isin(self.cycle, cycles)]
        
        if isinstance(cycles, str):
            # Handling condition-based filtering with regular expressions
            
            # Match '<n'
            if match := re.match(r'^<(\d+)$', cycles):
                return self[column][self.cycle < int(match.group(1))]
            
            # Match '>n'
            elif match := re.match(r'^>(\d+)$', cycles):
                return self[column][self.cycle > int(match.group(1))]
            
            # Match 'n:m' range
            elif match := re.match(r'^(\d+):(\d+)$', cycles):
                lower, upper = int(match.group(1)), int(match.group(2))
                return self[column][(self.cycle >= lower) & (self.cycle <= upper)]
            
            # Match 'n:' (from n to the end)
            elif match := re.match(r'^(\d+):$', cycles):
                lower = int(match.group(1))
                return self[column][self.cycle >= lower]
            
            # Match ':n' (from the start to n)
            elif match := re.match(r'^:(\d+)$', cycles):
                upper = int(match.group(1))
                return self[column][self.cycle <= upper]
            
            else:
                raise ValueError(f"Invalid condition string: {cycles}")
        
        raise TypeError("cycles must be an int, list, or str")

