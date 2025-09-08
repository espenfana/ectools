'''ChronoPotiometry class'''
from typing import Optional, Dict, Any, Union
from matplotlib.axes import Axes
import numpy as np

from .electrochemistry import ElectroChemistry
from ..utils import optional_return_figure

class ChronoPotentiometry(ElectroChemistry):
    '''Chronopotentiometry file container'''

    # Class variables and constants
    identifiers = {'Chronopotentiometry', 'CP', 'CHRONOP'}
    # Data columns to import
    column_patterns = {**ElectroChemistry.column_patterns,
    }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict
    
    # Type hints for technique-specific attributes
    time_step: float
    curr_step: float

    # Initialize
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        '''Create a Chronopotentiometry file container'''
        super().__init__(*args, **kwargs)
        # Set technique-specific metadata
        self.tag: str = 'CP'
        self.control: str = 'Amperostatic'
        # Note: time_step, curr_step are set during metadata parsing

    # Class methods
    def parse_meta_mpt(self) -> None:
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        self.curr_step = 'NotImplemented'  # TODO: Implement proper parsing

    def parse_meta_gamry(self) -> None:
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        # Note: time_step and curr_step are set during parsing
        # self.time_step and self.curr_step are assigned from metadata

    @optional_return_figure
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
        fig, ax = super().plot(ax=ax, x=x, y=y, mask=mask, hue=hue, 
                          add_aux_cell=add_aux_cell, add_aux_counter=add_aux_counter, 
                          ax_kws=ax_kws, return_figure=True, **kwargs)  # Always get the figure back from parent
        if hue:
            ax.legend(title=hue)
        return fig, ax
