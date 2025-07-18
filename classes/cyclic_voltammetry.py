'''Cyclic Voltammetry class'''
import re
from typing import Optional, Union, Dict, List, Any, Tuple
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from .electrochemistry import ElectroChemistry

# CyclicVoltammetry Class
class CyclicVoltammetry(ElectroChemistry):
    '''Cyclic voltammetry file container'''

    # Class variables and constants
    identifiers = {'Cyclic Voltammetry', 'CV'}  # Strings in the raw files which indicate the technique
    column_patterns = {**ElectroChemistry.column_patterns,
        'oxred': (r'ox/red',),
        'cat': (r'cat', r'cathodic'),
        'cycle': (r'cycle number', r'cycle'),
        }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.    
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict
    
    # Type hints for data columns (in addition to parent class)
    cycle: NDArray[np.int32]
    cycle_v2: NDArray[np.int32] 
    cycle_init: NDArray[np.int32]
    sweep_dir: NDArray[np.int8]
    oxred: NDArray[np.float64]
    cat: NDArray[np.float64]
    
    # Type hints for technique-specific metadata
    scanrate: float
    pot_init: float
    pot_upper: float
    pot_lower: float
    pot_end: float
    ncycles: int
    
    # Initialize
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        '''Create a Cyclic Voltammetry container'''
        super().__init__(*args, **kwargs)
        
        # Initialize data columns as empty arrays
        self.cycle = np.empty(0, dtype=np.int32)
        self.cycle_v2 = np.empty(0, dtype=np.int32)
        self.cycle_init = np.empty(0, dtype=np.int32)
        self.sweep_dir = np.empty(0, dtype=np.int8)  # -1 or 1
        self.oxred = np.empty(0, dtype=np.float64)
        self.cat = np.empty(0, dtype=np.float64)
        
        # Set technique-specific metadata
        self.tag: str = 'CV'
        self.control: str = 'Potentiostatic'
        
        # TODO: Updated to use dict for data_columns, extending with CV-specific columns
        self.data_columns.update({
            'cycle': 'Cycle Number (Source)',
            'cycle_v2': 'Cycle Number (Vertex)',
            'cycle_init': 'Cycle Number (Initial)',
            'sweep_dir': 'Sweep Direction',  # TODO Consider the use of these 3 columns
            'oxred': 'Oxidation/Reduction',
            'cat': 'Cathodic'
        })
        
        # Note: scanrate, pot_init, pot_upper, pot_lower, pot_end, ncycles are set during metadata parsing

    # Class methods
    def parse_meta_mpt(self) -> None:
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt()  # Preprocess the metadata block
        self.scanrate = float(self._meta_dict['dE/dt'][0][0])
        self.units['scanrate'] = self._meta_dict['dE/dt unit'][0]
        self.pot_init = float(self._meta_dict['Ei'][0][0])
        self.units['pot_init'] = self._meta_dict['Ei'][1]
        self.pot_upper = float(self._meta_dict['E1'][0][0])
        self.units['pot_upper'] = self._meta_dict['E1'][1]
        self.pot_lower = float(self._meta_dict['E2'][0][0])
        self.units['pot_lower'] = self._meta_dict['E2'][1]
        self.pot_end = float(self._meta_dict['Ef'][0][0])
        self.units['pot_end'] = self._meta_dict['Ef'][1]
        self.ncycles = int(self._meta_dict['nc cycles'][0][0])

    def parse_meta_gamry(self) -> None:
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        # tabsep gamry metadata is in meta_dict
        metamap = {'scanrate': 'SCANRATE', 
                   'pot_init': 'VINIT', 
                   'pot_upper':'VLIMIT1', 
                   'pot_lower':'VLIMIT2', 
                   'pot_end':'VFINAL', 
                   'ncycles': 'CYCLES'}
        for key, label in metamap.items():
            self[key] = float(self._meta_dict[label]['value'])
            self.units[key] = re.search(r'\((.*?)\)', self._meta_dict[label]['description']).group(1)
        self.ncycles = int(self.ncycles)
        if self.pot_upper < self.pot_lower:
            tmp = self.pot_lower
            self.pot_lower = self.pot_upper
            self.pot_upper = tmp

    def plot(self,
             ax: Optional[Axes] = None,
             x: str = 'pot',
             y: str = 'curr',
             hue: Optional[Union[str, bool]] = None,
             mask: Optional[np.ndarray] = None,
             add_aux_cell: bool = False,
             add_aux_counter: bool = False,
             ax_kws: Optional[Dict[str, Any]] = None,
             cycles: Optional[Union[List[int], Tuple[int, int]]] = None,
             **kwargs: Any) -> Axes:
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        if hue is True:  # default hue
            hue = 'cycle'
        if cycles:
            mask = np.where(np.logical_and(self['cycle']>=cycles[0], self['cycle']<=cycles[1]))
        ax = super().plot(ax=ax,
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
    
    def filter_cycle(self, column: str, cycles: Union[List[int], int, str]) -> np.ndarray:
        """
        Filter a data column based on cycle conditions.
        Parameters:
        column (str): The data column to filter.
        cycles (list | int | str): The cycle condition to filter by.
            - int: Single cycle number.
            - list: List of cycle numbers.
            - str: Condition string, e.g., '<10', '>5', '2:4', 'n:', ':n'.
        Returns:
        np.ndarray: Filtered data column.
        Raises:
        ValueError: If the condition string is invalid.
        TypeError: If cycles is not an int, list, or str.
        """
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
            if match := re.match(r'^>(\d+)$', cycles):
                return self[column][self.cycle > int(match.group(1))]

            # Match 'n:m' range
            if match := re.match(r'^(\d+):(\d+)$', cycles):
                lower, upper = int(match.group(1)), int(match.group(2))
                return self[column][(self.cycle >= lower) & (self.cycle <= upper)]

            # Match 'n:' (from n to the end)
            if match := re.match(r'^(\d+):$', cycles):
                lower = int(match.group(1))
                return self[column][self.cycle >= lower]

            # Match ':n' (from the start to n)
            if match := re.match(r'^:(\d+)$', cycles):
                upper = int(match.group(1))
                return self[column][self.cycle <= upper]

            raise ValueError(f"Invalid condition string: {cycles}")

        raise TypeError("cycles must be an int, list, or str")
