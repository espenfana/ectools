'''Linear sweep voltammetry container class'''
import re

from .electrochemistry import ElectroChemistry

class LinearSweepVoltammetry(ElectroChemistry):
    '''Linear sweep voltammetry file container'''

    # Class variables and constants
    identifiers = {'Linear Sweep Voltammetry', 'LSV'}
    column_patterns = {**ElectroChemistry.column_patterns,
        'oxred': (r'ox/red',),
        'cat': (r'cat', r'cathodic'),
        }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict

    # Initialize

    def __init__(self, *args, **kwargs):
        '''Create a Linear Sweep Voltammetry container'''
        self.scanrate = float()
        self.pot_init = float()
        self.pot_end = float()
        super().__init__(*args, **kwargs)

    # Class methods
    def parse_meta_mpt(self):
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        self.scanrate = float(self._meta_dict['dE/dt'][0][0])
        self.units['scanrate'] = self._meta_dict['dE/dt unit'][0]
        self.pot_init = float(self._meta_dict['Ei'][0][0])
        self.units['pot_init'] = self._meta_dict['Ei'][1]
        self.pot_end = float(self._meta_dict['Ef'][0][0])
        self.units['pot_end'] = self._meta_dict['Ef'][1]

    def parse_meta_gamry(self):
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        # tabsep gamry metadata is in meta_dict
        metamap = {'scanrate': 'SCANRATE', 'pot_init': 'VINIT', 'pot_end':'VFINAL'}
        for key, label in metamap.items():
            self[key] = float(self._meta_dict[label]['value'])
            self.units[key] = re.search(r'\((.*?)\)', self._meta_dict[label]['description']).group(1)

    def plot(self,
            ax=None,
            x='pot',
            y='curr',
            hue=None,
            mask=None,
            ax_kws=None,
            **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        ax = super().plot(
            ax=ax,
            x=x,
            y=y,
            mask=mask,
            hue=hue,
            ax_kws=ax_kws,
            **kwargs)
        if hue:
            ax.legend(title=hue)
        return ax
