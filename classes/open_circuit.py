'''Open Circuit class'''
from typing import Optional, Dict, Any, Union
from matplotlib.axes import Axes
import numpy as np

from .electrochemistry import ElectroChemistry

class OpenCircuit(ElectroChemistry):
    '''Open Circuit Voltage file container'''

    # Class variables and constants
    identifiers = {'Open Circuit', 'CORPOT'}
    column_patterns = {**ElectroChemistry.column_patterns,
    }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict

    #Initialize
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        '''Create a Open Circuit type file container'''
        super().__init__(*args, **kwargs)
        self.tag: str = 'OCP'
        self.control: str = 'Open Circuit'

    # Class methods
    #def parse_meta_mpt(self):
    #    '''Parse the metadata blocks into attributes'''
    #    super().parse_meta_mpt() # Preprocess the metadata block

    def plot(self,
            ax: Optional[Axes] = None,
            x: str = 'time',
            y: str = 'pot',
            hue: Optional[Union[str, bool]] = None,
            mask: Optional[np.ndarray] = None,
            add_aux_cell: bool = False,
            add_aux_counter: bool = False,
            ax_kws: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Axes:
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        kws = {'xlabel': f'time ({self.units[x]})',
                'ylabel': f'E_OC ({self.units[y]})'}
        ax = super().plot(ax=ax, x=x, y=y, mask=mask, hue=hue, 
                          add_aux_cell=add_aux_cell, add_aux_counter=add_aux_counter,
                          ax_kws=kws, **kwargs)
        if hue:
            ax.legend(title=hue)
        return ax
