'''ChronoAmperometry class'''
from typing import Optional, Dict, Any, Union
from matplotlib.axes import Axes
import numpy as np

from .electrochemistry import ElectroChemistry
from ..utils import optional_return_figure

class ChronoAmperometry(ElectroChemistry):
    '''Chronoamperometry file container'''

    # Class variables and constants
    _identifiers = {'Chronoamperometry', 'CHRONOA'} # Strings in the files which indicate technique
    _column_patterns = {**ElectroChemistry._column_patterns,
    }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict

    # Initialize
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        '''Create a Chronoamperometry file container'''
        super().__init__(*args, **kwargs)
        # Set technique-specific metadata
        self.tag: str = 'CA'
        self.control: str = 'Potentiostatic'

    # Class methods
    def parse_meta_mpt(self) -> None:
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        

    @optional_return_figure
    def plot(self,
            ax: Optional[Axes] = None,
            x: str = 'time',
            y: str = 'curr',
            hue: Optional[Union[str, bool]] = None,
            mask: Optional[np.ndarray] = None,
            add_aux_cell: bool = False,
            add_aux_counter: bool = False,
            ax_kws: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Axes:
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        fig, ax = super().plot(ax=ax, x=x, y=y, mask=mask, hue=hue, 
                         add_aux_cell=add_aux_cell, add_aux_counter=add_aux_counter,
                         ax_kws=ax_kws, return_figure=True, **kwargs)  # Always get the figure back from parent
        if hue:
            ax.legend(title=hue)
        return fig, ax
