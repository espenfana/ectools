'''Linear sweep voltammetry container class'''
import re
from typing import Optional, Dict, Any, Union
from matplotlib.axes import Axes
import numpy as np

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
    
    # Type hints for technique-specific metadata
    scanrate: float
    pot_init: float
    pot_end: float

    # Initialize
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        '''Create a Linear Sweep Voltammetry container'''
        super().__init__(*args, **kwargs)
        # Set technique-specific metadata
        self.tag: str = 'LSV'
        self.control: str = 'Potentiostatic'
        # Note: scanrate, pot_init, pot_end are set during metadata parsing

    # Class methods
    def parse_meta_mpt(self) -> None:
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        self.scanrate = float(self._meta_dict['dE/dt'][0][0])
        self.units['scanrate'] = self._meta_dict['dE/dt unit'][0]
        self.pot_init = float(self._meta_dict['Ei'][0][0])
        self.units['pot_init'] = self._meta_dict['Ei'][1]
        self.pot_end = float(self._meta_dict['Ef'][0][0])
        self.units['pot_end'] = self._meta_dict['Ef'][1]

    def parse_meta_gamry(self) -> None:
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        # tabsep gamry metadata is in meta_dict
        metamap = {'scanrate': 'SCANRATE', 'pot_init': 'VINIT', 'pot_end':'VFINAL'}
        for key, label in metamap.items():
            self[key] = float(self._meta_dict[label]['value'])
            self.units[key] = re.search(r'\((.*?)\)', self._meta_dict[label]['description']).group(1)

    def plot(self,
            ax: Optional[Axes] = None,
            x: str = 'pot',
            y: str = 'curr',
            hue: Optional[Union[str, bool]] = None,
            mask: Optional[np.ndarray] = None,
            add_aux_cell: bool = False,
            add_aux_counter: bool = False,
            ax_kws: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Axes:
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        ax = super().plot(
            ax=ax,
            x=x,
            y=y,
            mask=mask,
            hue=hue,
            add_aux_cell=add_aux_cell,
            add_aux_counter=add_aux_counter,
            ax_kws=ax_kws,
            **kwargs)
        if hue:
            ax.legend(title=hue)
        return ax
