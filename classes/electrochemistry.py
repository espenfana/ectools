'''Parent electrochemistry file class'''

import re
from datetime import datetime, timedelta
from typing import Union, Tuple, Optional, Dict, List, Any
import warnings

import dateutil.parser as date_parser
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Conditional import of Bokeh
try:
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import factor_cmap
except ImportError:
    pass

from ..config import requires_bokeh, bokeh_conf, LOCAL_TZ

class ElectroChemistry():
    ''' The default container and parent class for containing electrochemistry files and methods
    '''
    # Class variables and constants
    identifiers = set()

    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.search.
    column_patterns = {
        'redherring': (r'redherring',), # An identifier which is not found will not generate errors
        'signal': (r'Sig',), # Signal, i.e. target potenital
        'time': (r'time/(.?s)',r'^T$',), # Time column
        'pot': (r'<?Ewe>?/(.?V)', r'potential', r'^Vf$',), # Potential column
        'curr':(r'<?I>?/(.?A)', r'^Im$')} # Current column
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict
    
    # Type hints for data columns
    time: np.ndarray
    curr: np.ndarray
    curr_dens: np.ndarray
    pot: np.ndarray
    timestamp: np.ndarray
    
    # Type hints for metadata and other attributes
    fname: str
    fpath: str
    meta: List[str]
    tag: Optional[str]
    control: Optional[str]
    data_columns: List[str]
    units: Dict[str, str]
    aux: Dict[str, Dict[str, Any]]
    area: float
    starttime: datetime
    starttime_toffset: float
    label: Optional[str]
    we_number: Optional[int]
    def __init__(self, fname: str, fpath: str, meta: List[str], **kwargs: Any):
        ''' Create a generalized ElecroChemistry object'''
        # Container metadata
        self.tag: Optional[str] = None
        self.control: Optional[str] = None
        self.fname: str = fname # Filename
        self.fpath: str = fpath # Path to file
        self.meta: List[str] = meta # Metadata block
        
        # Apply any additional keyword arguments
        for key, val in kwargs.items():
            setattr(self, key, val)
        # Initialize data columns as empty arrays
        self.time = np.empty(0)
        self.curr = np.empty(0)
        self.curr_dens = np.empty(0)
        self.pot = np.empty(0)
        self.timestamp = np.empty(0)
        self.data_columns = ['time', 'curr', 'curr_dens', 'pot', 'timestamp']
        self.units = {}
        self._meta_dict = {}
        # These should remain empty in this class
        #self.cycle = np.empty(0)
        #self.oxred = np.empty(0)
        self.aux = {'pico': {}, 'furnace': {}} # Auxiliary data
        # Note: area, starttime, starttime_toffset are set during file parsing
        # No need to initialize them here since they're assigned complete values
        self.label = None # Used for automatic labeling of plots
        self._potential_offset = 0.0  # Initialize potential offset to zero
        self.we_number = None # Working electrode number

    def __getitem__(self, key):
        '''Makes object subscriptable like a dict'''
        return self.__getattribute__(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        '''Makes object attributes assignable like a dict'''
        self.__setattr__(key, value)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} object from file {self.fname}'
    
    def __contains__(self, key):
        '''Check if attribute exists (for use with 'in' operator)'''
        return hasattr(self, key)

    # Data parsing methods
    # ------------------------

    def parse_meta_mpt(self):
        '''Parse attributes from the metadata block'''
        #self.colon_delimited = [row.split(':') for row in self.meta]
        colonsep = {} # For colon sepatated metadata
        wsep = [] # For width separated metadata, which may include
        # 20 pt splitter and stripper replaced by function split_by_length
        #split20 = lambda s: [s.strip()] if len(s) < 20 else [s[:20].strip(), *split20(s[20:])]
        for row in self.meta:
            if (':' in row) and (row[18:20] != '  '):
                colonsep[row.split(':', 1)[0].strip()] = row.split(':', 1)[1].strip()
            elif (len(row)>20) and (row[18:20] == '  '):
                if re.match(r'vs\.', row):
                    wsep[-1].append(_split_by_length(row[20:]))
                else:
                    m = re.search(r'\((.?\w)\)', row[:20]) # look for units in parenthesis
                    if m:
                        wsep.append([row[:m.start(0)].strip(),
                                     _split_by_length(row[20:]), m.group(1)])
                    else:
                        wsep.append([row[:20].strip(), 
                                     _split_by_length(row[20:])])
        # If the experiment was modified during run, the last value will be entered
        widthsep = {row[0]: row[1:] for row in wsep}
        self._meta_dict = {**colonsep, **widthsep}
        self.starttime = date_parser.parse(self._meta_dict['Acquisition started on'])

    def parse_meta_gamry(self):
        '''parse attribudes from the metadata block'''
        i = 0
        while i < len(self.meta):
            line = self.meta[i]
            match len(line):
                case 2:
                    self._meta_dict[line[0]] = {'value': line[1]}
                case 4 | 5:
                    if line[0] == 'NOTES': # handle the multi-line note field
                        self._meta_dict[line[0]] = {'value': ""}
                        i += 1
                        for _ in range(i,int(line[2])):
                            self._meta_dict[line[0]]['value'] += self.meta[i]
                            i += 1
                    else:
                        self._meta_dict[line[0]] = {
                            'label': line[1],
                            'value': line[2],
                            'description': ', '.join(line[3:])
                        }
            i += 1
        # Import into attributes
        metamap = {'area': 'AREA'}
        for key, label in metamap.items():
            try:
                self[key] = float(self._meta_dict[label]['value'])
                self.units[key] = re.search(r'\((.*?)\)',
                                            self._meta_dict[label]['description']).group(1)
            except (KeyError, IndexError):
                pass

        date_str = self._meta_dict['DATE']['value']
        time_str = self._meta_dict['TIME']['value']
        self.starttime = date_parser.parse(date_str + ' ' + time_str).replace(tzinfo=LOCAL_TZ)
        # Absulute time calculation. Needs to account for (1) pre step and (2) ocv delay time
        if self.time[0] < 0:
            self.starttime_toffset = float(self._meta_dict['TPRESTEP']['value'])
        else:
            self.starttime_toffset = 0
        self.starttime_toffset += float(getattr(self, 'ocv_delay_time', 0))
        timedeltas = np.array([timedelta(seconds=t + self.starttime_toffset) for t in self.time])
        self.timestamp = np.array([self.starttime + delta for delta in timedeltas])

    # Data manipulation methods
    # ------------------------

    def get_data_dict(self) -> dict:
        """
        Get all data columns as a dictionary.
        """
        return {key: self[key] for key in self.data_columns}

    @property
    def pot_offset(self):
        """
        Return the 'pot' values offset by the potential offset.
        """
        return self.pot + self._potential_offset

    def set_pot_offset(self, offset):
        """
        Set the potential offset value.

        Args:
            offset (float): The offset value in volts.
        """
        if not isinstance(offset, (int, float)):
            raise ValueError("Offset must be a numeric value.")
        if offset < -2 or offset > 2:
            raise ValueError("Offset must be between -2 and 2 volts.")
        self._potential_offset = offset
        if 'pot_offset' not in self.data_columns:
            self.data_columns.append('pot_offset')

    def slice(self, **criteria: Union[float, Tuple[float, float]]) -> 'ElectroChemistry':
        """
        Return a sliced version of the instance based on the specified criteria.

        Args:
            **criteria: Keyword arguments where keys are data column names and values are
                        either a single scalar or a tuple specifying (lower, upper) limits.

        Returns:
            ElectroChemistry: A new instance of the class with sliced data.

        Raises:
            ValueError: If an invalid data column is specified or if the value is improperly formatted.
        """
        # Initialize the mask to all True
        mask = np.ones(len(self.time), dtype=bool)

        # Iterate over the criteria to build the mask
        for key, value in criteria.items():
            if key not in self.data_columns:
                raise ValueError(f"Data column '{key}' not found in the data columns.")

            data_column = self[key]

            # Determine the type of slicing based on the value
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"The value for '{key}' must be a single value or a tuple of length 2.")
                lower, upper = value
                criterion_mask = (data_column >= lower) & (data_column <= upper)
            else:
                criterion_mask = (data_column == value)

            # Combine the masks
            mask = mask & criterion_mask
            if not np.any(mask):
                warnings.warn(f"No elements match the slicing criteria: {criteria}. Returning an empty instance.")

        # Create a new, uninitialized instance of the same class
        sliced_instance = self.__class__.__new__(self.__class__)

        # Copy over all attributes
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == mask.shape[0]:
                # Slice data arrays
                setattr(sliced_instance, attr_name, attr_value[mask])
            else:
                # Copy other attributes directly
                if attr_name == 'aux':
                    continue
                setattr(sliced_instance, attr_name, attr_value)

        # Configure auxiliary channels
        # Warning: auxiliary data handling not generalized
        if self.aux is not None:
            sliced_instance.aux = {}
            for attr_name, attr_value in self.aux.items():
                if attr_name == 'pico':
                    # Handle pico data, checking if value shape matches mask
                    sliced_instance.aux['pico'] = {}
                    for key, value in attr_value.items():
                        if isinstance(value, np.ndarray) and len(value) == len(mask):
                            sliced_instance.aux['pico'][key] = value[mask]
                        else:
                            sliced_instance.aux['pico'][key] = value
                elif attr_name == 'furnace': # Furnace data not interpolated to common time axis
                    sliced_instance.aux['furnace'] = {key: value for key, value in attr_value.items()}
                else:
                    sliced_instance.aux[attr_name] = attr_value
        else:
            sliced_instance.aux = None
        return sliced_instance

    # Output methods
    # ------------------------
    def to_csv(self, fname: str = None) -> None:
        """
        Save the data to a CSV file.
        If fname is not provided, it will use the instance's fname attribute.
        """
        if fname is None:
            fname = self.fname
        if not fname.endswith('.csv'):
            fname += '.csv'
        
        # Create a DataFrame from the data dictionary
        import pandas as pd
        df = pd.DataFrame(self.get_data_dict())
        
        # Save to CSV
        df.to_csv(fname, index=False)

    # Plotting related methods
    # ------------------------

    def makelab(self, key):
        '''Generate an axis label with unit'''
        d = {'curr': 'I ', 'pot': 'E ', 'time': 't', 'curr_dens': 'I\''}
        if key not in d:
            return key
        label = d[key]
        if key in self.units:
            label += f' ({self.units[key]})'
        return label

    @requires_bokeh
    def plot_bokeh(self, x='time', y='curr', hue=None, title=None): # Added hue parameter
        """Plot using Bokeh with global settings."""
        title = title or self.fname
        # Use Bokeh settings for figure size, title, and tooltips

        p = figure(
            width=bokeh_conf.figsize[0],  # Use the configured figure size
            height=bokeh_conf.figsize[1],
            title=title,
        )
        p.add_tools(self.get_hover_tool())
        if hue is None:
            color = 'blue'  # Default color
        else:
            unique_hues = list(set(self[hue]))  # Get unique values in the hue column
            # Create a color mapper
            color = factor_cmap(hue, palette='Category10_10', factors=unique_hues)

        p.line(x, y,
            source=ColumnDataSource(self.get_data_dict()),
            color=color)
        show(p)

    @requires_bokeh
    def get_hover_tool(self) -> 'HoverTool':
        '''Get hover tooltips for Bokeh plot'''
        hover = HoverTool()
        tooltips = []
        for key in self.data_columns:
            if key != 'timestamp':
                tooltips.append((self.makelab(key), f'@{key}'))

        tooltips.append(('timestamp', '@timestamp{%F %T}'))
        hover.tooltips = tooltips
        hover.formatters = {'@timestamp': 'datetime'}
        return hover

    def plot(self,
        ax=None, # pyplot axes
        x='time', # key for x axis array
        y='curr', # key for y axis array
        hue = None, # split the plot based on values in a third array
        mask = None, # logical array to slice the arrays
        add_aux_cell = False, # add auxiliary cell potential
        add_aux_counter = False, # add auxiliary counter potential
        ax_kws = None, # arguments passed to ax.set()
        **kwargs):
        '''Plot data using matplotlib.
            Parameters are seaborn-like. Any additional kwargs are passed along to pyplot'''
        ax_kws = ax_kws or {}
        if x not in ('time', 'timestamp'):
            add_aux_cell = False
            add_aux_counter = False
        if not ax:
            _, ax = plt.subplots()
        if not mask:
            mask = np.full(self[x].shape, True)
        if hue is True: # no hue set by technique
            hue = False
        if hue:
            for val in np.unique(self[hue][mask]):
                ax.plot(
                    self[x][self[hue]==val],
                    self[y][self[hue]==val],
                    label = str(val),
                     **kwargs)
        else:
            if 'label' not in kwargs:
                kwargs['label'] = self.label
            ax.plot(
                self[x][mask],
                self[y][mask],
                **kwargs)
        if add_aux_cell:
            if np.any(np.isfinite(self.aux['pico']['pot'])):
                last_color = ax.lines[-1].get_color()
                ax.plot(
                    self.aux['pico'][x],
                    self.aux['pico']['pot'],
                    label='Cell potential',
                    color=last_color, alpha=0.5)
            else:
                warnings.warn('No auxiliary cell potential data found')
        if add_aux_counter:
            if np.any(np.isfinite(self.aux['pico']['pot'])):
                last_color = ax.lines[-1].get_color()
                ax.plot(
                    self.aux['pico'][x],
                    self.aux['pico']['counter_pot'],
                    label='Counter potential',
                    color=last_color, alpha=0.5)
            else:
                warnings.warn('No auxiliary counter potential data found')
        if 'xlabel' not in ax_kws:
            ax_kws['xlabel'] = self.makelab(x)
        if 'ylabel' not in ax_kws:
            ax_kws['ylabel'] = self.makelab(y)
        ax.set(**ax_kws) # Set axes properties, such as xlabel etc.
        ax.legend()
        return ax

    def plotyy(self,
            fig = None,
            x = 'time', # key for the common x-axis
            y_left = 'pot', # key for left y-axis
            color_left = 'tab:blue', # color for left y-axis
            y_right = 'curr', # key for right y-axis
            color_right = 'tab:red', # color for right y-axis
            hue = None, # split the plot based on values in a third array
            mask = None, # logical array to slice the arrays
            ax_left_kws = None, # arguments passed to ax.set()
            ax_right_kws = None, # arguments passed to ax.set()
            **kwargs):
        ''' Possibly not properly implemented!
        Plot data with two y-scales using matplotlib. 
            Parameters are seaborn-like. Any additional kwargs are passed along to pyplot'''
        if not fig:
            fig = plt.figure()
        ax_left = fig.add_subplot(111)
        ax_left = self.plot(ax=ax_left,
                            x=x,
                            y=y_left,
                            color=color_left,
                            hue=hue,
                            mask=mask,
                            ax_kws=ax_left_kws,
                            **kwargs)

        ax_right = ax_left.twinx()
        ax_right = self.plot(ax=ax_right,
                             x=x,
                             y=y_right,
                             color=color_right,
                             hue=hue,
                             mask=mask,
                             ax_kws=ax_right_kws,
                             **kwargs)

        ax_left.spines['left'].set_color(color_left)
        ax_right.spines['right'].set_color(color_right)
        return fig, (ax_left, ax_right)

    def set_area(self, new_area: float):
        '''Set new value for area and recalculate current density
            new_area: float [cm²]'''
        self.area = new_area
        if self.curr.size > 0:
            self.curr_dens = self.curr / new_area
            self.units['curr_dens'] = f'{self.units["curr"]}/cm²'


    def pot_corrected(self, correction):
        """
        Return the potential column corrected by the specified value.
        Args:
            correction (float): The correction value in volts.
        Returns:
            np.ndarray: Corrected potential values.
        """
        return self.pot + correction

def _split_by_length(s, length=20):
    """
    Splits the input string `s` into chunks of `length`, stripping whitespace from each chunk.
    
    Parameters:
    - s: The input string to be split.
    - length: The length of each chunk (default is 20).
    
    Returns:
    - A list of stripped chunks of the input string.
    """
    result = []
    while s:
        result.append(s[:length].strip())
        s = s[length:]
    return result
