'''Parent electrochemistry file class'''

import re
from datetime import datetime, timedelta
from typing import Union, Tuple, Optional, Dict, List, Any, TYPE_CHECKING, cast
import warnings

import dateutil.parser as date_parser
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Only for type checking, not runtime
    pass

# Conditional import of Bokeh
try:
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import factor_cmap
except ImportError:
    pass

from ..config import requires_bokeh, bokeh_conf, LOCAL_TZ
from ..utils import optional_return_figure, split_by_length


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
    
    # Type hints for auxiliary data columns (dynamically added by auxiliary sources)
    # These are Optional because they're only present when auxiliary data sources are available
    cell_pot: Optional[np.ndarray]  # Added by PicoLogger auxiliary source
    counter_pot: Optional[np.ndarray]  # Calculated from cell_pot and pot
    cascade_temperature: Optional[np.ndarray]  # Added by FurnaceLogger auxiliary source
    cascade_rate: Optional[np.ndarray]  # Added by FurnaceLogger auxiliary source
    
    # Type hints for metadata and other attributes
    fname: str
    fpath: str
    meta: List[str]
    tag: Optional[str]
    control: Optional[str]
    data_columns: Dict[str, str]
    units: Dict[str, str]
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
        self.data_columns = { # Expected data columns, format: short_name: Display name (Unit)
            'time': 'Time (s)',
            'curr': 'Current (A)',
            'curr_dens': 'Current Density (A/cm²)',
            'pot': 'Potential (V)',
            'timestamp': 'Timestamp'
            # TODO: Consider adding 'signal' from column_patterns - check if this should be included

        }
        self.units = {}
        self._meta_dict = {}
        # These should remain empty in this class
        #self.cycle = np.empty(0)
        #self.oxred = np.empty(0)
        # Note: area, starttime, starttime_toffset are set during file parsing
        # No need to initialize them here since they're assigned complete values
        self.label = None # Used for automatic labeling of plots
        self._potential_offset = 0.0  # Initialize potential offset to zero
        self.we_number = None # Working electrode number

    @property
    def aux(self):
        """
        Deprecated: The 'aux' attribute is no longer used in the new auxiliary framework.
        
        Auxiliary data is now directly available as data columns on the main object.
        For example:
        - Instead of: obj.aux['pico']['cell_pot']
        - Use: obj.cell_pot
        
        Available auxiliary columns can be found in obj.data_columns dictionary.
        """
        import warnings
        available_aux_columns = [col for col in self.data_columns.keys() 
                               if col not in ['time', 'curr', 'curr_dens', 'pot', 'timestamp']]
        
        if available_aux_columns:
            aux_list = ', '.join(available_aux_columns)
            message = (
                f"The 'aux' attribute is deprecated. Auxiliary data is now available as direct attributes.\n"
                f"Available auxiliary columns: {aux_list}\n"
                f"Example: instead of obj.aux['pico']['cell_pot'], use obj.cell_pot"
            )
        else:
            message = (
                "The 'aux' attribute is deprecated. No auxiliary data columns found on this object.\n"
                "Auxiliary data is now integrated as direct attributes when available."
            )
            
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return None
    
    @aux.setter
    def aux(self, value):
        """Prevent setting the deprecated aux attribute."""
        import warnings
        warnings.warn(
            "Setting 'aux' attribute is deprecated. Auxiliary data is now managed automatically "
            "through the new auxiliary framework and stored as direct attributes.",
            DeprecationWarning, 
            stacklevel=2
        )

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
                    wsep[-1].append(split_by_length(row[20:]))
                else:
                    m = re.search(r'\((.?\w)\)', row[:20]) # look for units in parenthesis
                    if m:
                        wsep.append([row[:m.start(0)].strip(),
                                     split_by_length(row[20:]), m.group(1)])
                    else:
                        wsep.append([row[:20].strip(), 
                                     split_by_length(row[20:])])
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
        # TODO: Updated to use dict keys instead of list
        return {key: self[key] for key in self.data_columns.keys()}

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
        # TODO: Updated to use dict for data_columns
        if 'pot_offset' not in self.data_columns:
            self.data_columns['pot_offset'] = 'Offset Potential (V)'

    def slice(self, mask=None, **criteria) -> 'ElectroChemistry':
        """
        Return a sliced version of the instance based on the specified criteria or mask.

        Args:
            mask : array-like, optional
                Boolean mask to apply directly. If provided, criteria are ignored.
            **criteria: Keyword arguments where keys are data column names and values can be:
                - Single value for equality (e.g., substep=1)
                - Tuple (lower, upper) for range (e.g., time=(10, 100))
                - Dict with operator and value (e.g., pot={'>=': 0.5, '<': 1.0})

        Returns:
            ElectroChemistry: A new instance with extracted data (clean numpy arrays).

        Raises:
            ValueError: If an invalid data column is specified or if the value is improperly formatted.
            
        Examples:
            obj.slice(substep=1)  # Equal to 1
            obj.slice(time=(10, 100))  # Between 10 and 100
            obj.slice(pot={'>': 0.5})  # Greater than 0.5
            obj.slice(curr={'>=': -0.1, '<=': 0.1})  # Multiple conditions
            obj.slice(mask=my_boolean_array)  # Direct mask
        """
        # Handle direct mask or build mask from criteria
        if mask is not None:
            # Use provided mask directly
            if len(mask) != len(self.time):
                raise ValueError(f"Mask length ({len(mask)}) doesn't match data length ({len(self.time)})")
            final_mask = np.array(mask, dtype=bool)
        else:
            # Build mask from criteria
            final_mask = np.ones(len(self.time), dtype=bool)

            # Iterate over the criteria to build the mask
            for key, value in criteria.items():
                if key not in self.data_columns.keys():
                    raise ValueError(f"Data column '{key}' not found in the data columns.")

                data_column = self[key]

                # Handle different value types
                if isinstance(value, dict):
                    # Dictionary with operators: {'>=': 0.5, '<': 1.0}
                    for op, threshold in value.items():
                        if op == '>=':
                            criterion_mask = data_column >= threshold
                        elif op == '>':
                            criterion_mask = data_column > threshold
                        elif op == '<=':
                            criterion_mask = data_column <= threshold
                        elif op == '<':
                            criterion_mask = data_column < threshold
                        elif op == '==':
                            criterion_mask = data_column == threshold
                        elif op == '!=':
                            criterion_mask = data_column != threshold
                        else:
                            raise ValueError(f"Unsupported operator '{op}'. Use: '>=', '>', '<=', '<', '==', '!='")
                        final_mask = final_mask & criterion_mask
                elif isinstance(value, tuple):
                    # Tuple for range: (lower, upper)
                    if len(value) != 2:
                        raise ValueError(f"The value for '{key}' must be a single value, tuple of length 2, or dict with operators.")
                    lower, upper = value
                    criterion_mask = (data_column >= lower) & (data_column <= upper)
                    final_mask = final_mask & criterion_mask
                else:
                    # Single value for equality
                    criterion_mask = (data_column == value)
                    final_mask = final_mask & criterion_mask

            if not np.any(final_mask):
                warnings.warn(f"No elements match the slicing criteria: {criteria}. Returning an empty instance.")

        # Create a new, uninitialized instance of the same class
        sliced_instance = self.__class__.__new__(self.__class__)

        # Copy over all attributes, using boolean indexing for clean data extraction
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == final_mask.shape[0]:
                # Use boolean indexing to extract only matching data
                # This gives clean numpy arrays that work seamlessly with matplotlib
                # Perfect for: x = obj.slice(substep=1).timestamp; ax.plot(x, y)
                extracted_array = attr_value[final_mask]
                setattr(sliced_instance, attr_name, extracted_array)
            else:
                setattr(sliced_instance, attr_name, attr_value)
        
        return sliced_instance

    # Output methods
    # ------------------------
    def to_csv(self, fname: str = None) -> None:
        """
        Save all data columns to a CSV file.
        
        This method automatically includes all available data columns, including
        any auxiliary data that has been integrated through the auxiliary framework.
        
        Args:
            fname: Output filename. If not provided, uses the instance's fname attribute.
        """
        if fname is None:
            fname = self.fname
        if not fname.endswith('.csv'):
            fname += '.csv'
        
        # Create a DataFrame from all available data columns
        import pandas as pd
        df = pd.DataFrame(self.get_data_dict())
        
        # Remap column headers to display names
        df.rename(columns=self.data_columns, inplace=True)
        
        # Save to CSV
        df.to_csv(fname, index=False)

    # Plotting related methods
    # ------------------------

    def makelab(self, key):
        '''Generate an axis label with unit'''
        # TODO: Consider using display names from data_columns dict instead of hardcoded labels
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
        # TODO: Updated to use dict keys instead of list
        for key in self.data_columns.keys():
            if key != 'timestamp':
                tooltips.append((self.makelab(key), f'@{key}'))

        tooltips.append(('timestamp', '@timestamp{%F %T}'))
        hover.tooltips = tooltips
        hover.formatters = {'@timestamp': 'datetime'}
        return hover

    @optional_return_figure
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
            Parameters are seaborn-like. Column filtering can be done via kwargs (e.g., substep=1, cycle=2).
            Use "-" prefix (e.g., y="-curr") to flip sign. Other kwargs are passed along to pyplot.
            
            Args:
                return_figure: If True, return figure/axes. If False, show plot and return None.
                
            Returns:
                If return_figure=True: (fig, ax) if new figure created, or just ax if axes provided.
                If return_figure=False: None (after showing plot).
            '''
        ax_kws = ax_kws or {}
        if x not in ('time', 'timestamp'):
            add_aux_cell = False
            add_aux_counter = False
        
        user_provided_ax = ax is not None
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = None
        
        # Separate column filters from plot styling kwargs
        column_filters = {}
        plot_kwargs = {}
        
        for key, value in kwargs.items():
            if hasattr(self, key) and hasattr(getattr(self, key), '__len__'):
                # This is likely a data column that can be used for filtering
                column_filters[key] = value
            else:
                # This is a plot styling parameter
                plot_kwargs[key] = value
        
        # Build combined mask from mask parameter and column filters
        if not mask:
            mask = np.full(self[x].shape, True)
        else:
            mask = mask.copy()  # Don't modify the original mask
        
        # Apply column filtering
        filter_info = []
        for col_name, filter_value in column_filters.items():
            try:
                col_data = getattr(self, col_name)
                if hasattr(col_data, '__len__') and len(col_data) == len(mask):
                    col_mask = col_data == filter_value
                    mask = mask & col_mask
                    filter_info.append(f"{col_name}={filter_value}")
                else:
                    warnings.warn(f"Column '{col_name}' exists but has incompatible shape for filtering")
            except AttributeError:
                warnings.warn(f"Column filtering requested ({col_name}={filter_value}) but '{col_name}' column not found in data")
        
        # Handle sign flipping with '-' prefix
        flip_y = False
        y_column = y
        if y.startswith('-'):
            flip_y = True
            y_column = y[1:]  # Remove the '-' prefix to get actual column name
        
        # Get data and apply sign flipping if requested
        x_data = self[x][mask]
        y_data = self[y_column][mask]
        if flip_y:
            y_data = -y_data
        
        if hue is True: # no hue set by technique
            hue = False
        if hue:
            for val in np.unique(self[hue][mask]):
                hue_mask = self[hue][mask] == val
                y_hue_data = self[y_column][self[hue]==val]
                if flip_y:
                    y_hue_data = -y_hue_data
                ax.plot(
                    self[x][self[hue]==val],
                    y_hue_data,
                    label = str(val),
                     **plot_kwargs)
        else:
            if 'label' not in plot_kwargs:
                plot_kwargs['label'] = self.label
            
            # Use numpy masked arrays to hide unwanted values while preserving structure
            x_plot = np.ma.array(self[x], mask=~mask)
            y_plot = np.ma.array(self[y_column], mask=~mask)
            if flip_y:
                y_plot = np.ma.multiply(y_plot, -1)
            
            # Suppress the masked array conversion warning during plotting
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan')
                ax.plot(
                    x_plot,
                    y_plot,
                    **plot_kwargs)
        if add_aux_cell:
            if hasattr(self, 'cell_pot') and np.any(np.isfinite(self.cell_pot)):
                last_color = ax.lines[-1].get_color()
                ax.plot(
                    x_data,  # Use the pre-extracted x-axis data
                    self.cell_pot[mask],  # pylint: disable=unsubscriptable-object
                    label='Cell potential',
                    color=last_color, alpha=0.5)
            else:
                warnings.warn('No auxiliary cell potential data found')
        if add_aux_counter:
            if hasattr(self, 'counter_pot') and np.any(np.isfinite(self.counter_pot)):
                last_color = ax.lines[-1].get_color()
                ax.plot(
                    x_data,  # Use the pre-extracted x-axis data
                    self.counter_pot[mask],  # pylint: disable=unsubscriptable-object
                    label='Counter potential',
                    color=last_color, alpha=0.5)
            else:
                warnings.warn('No auxiliary counter potential data found')
        if 'xlabel' not in ax_kws:
            ax_kws['xlabel'] = self.makelab(x)
        if 'ylabel' not in ax_kws:
            y_label = self.makelab(y_column)
            if flip_y:
                y_label = f"- {y_label}"
            ax_kws['ylabel'] = y_label
        ax.set(**ax_kws) # Set axes properties, such as xlabel etc.
        ax.legend()
        
        # If user provided ax, they're managing their own figure - return just the axes
        # If we created the figure, return both for the decorator to handle
        if user_provided_ax:
            return ax
        else:
            return fig, ax

    @optional_return_figure
    def plot_dual_y(self,
                   x='time',  # key for the common x-axis
                   y_left='pot',  # key for left y-axis
                   y_right='curr',  # key for right y-axis
                   color_left='tab:blue',  # color for left y-axis
                   color_right='tab:orange',  # color for right y-axis
                   figsize=(10, 6),
                   mask=None,  # logical array to slice the arrays
                   # Advanced scaling options (for similar-unit dual plots like dual potentials)
                   scale_range=None,  # Enable matched scaling: None=off, 'auto'=auto-calc, float=fixed range
                   percentile_range=(5, 95),  # Percentile range to exclude outliers
                   tick_spacing=0.05,  # Tick spacing for matched scaling
                   # Standard plot options
                   hue=None,  # split the plot based on values in a third array
                   ax_left_kws=None,  # arguments passed to left ax.set()
                   ax_right_kws=None,  # arguments passed to right ax.set()
                   **kwargs):
        """
        Plot data with two y-scales using matplotlib with optional advanced scaling.
        
        Parameters:
        -----------
        x, y_left, y_right : str
            Keys for data columns to plot. Use "-" prefix (e.g., "-pot") to flip sign
        color_left, color_right : str
            Colors for left and right y-axes
        figsize : tuple
            Figure size
        mask : array-like, optional
            Boolean mask to filter data
        scale_range : None, 'auto', or float, default=None
            Matched scaling mode: None=independent axes, 'auto'=auto-calculated range, 
            float=fixed range (e.g., 0.5 for 0.5V range)
        percentile_range : tuple, default=(5, 95)
            Percentile range for outlier exclusion when using matched scaling
        tick_spacing : float, default=0.05
            Tick spacing when using matched scaling
        hue : str, optional
            Column for color grouping
        ax_left_kws, ax_right_kws : dict, optional
            Additional axis formatting arguments
        return_figure : bool, default=False
            If True, return (fig, (ax_left, ax_right)) for manual handling.
            If False, automatically call plt.show() and return None.
        **kwargs : dict
            Column filtering (e.g., substep=1, cycle=2) and plot styling arguments
            
        Returns:
        --------
        fig, (ax_left, ax_right) : matplotlib figure and axes objects
            Only returned if return_figure=True. Otherwise returns None after showing plot.
            Only returned if auto_show=False. Otherwise returns None after showing plot.
        """
        ax_left_kws = ax_left_kws or {}
        ax_right_kws = ax_right_kws or {}
        
        # Create figure and axes
        fig, ax_left = plt.subplots(figsize=figsize)
        ax_right = ax_left.twinx()
        
        # Separate column filters from plot styling kwargs
        column_filters = {}
        plot_kwargs = {}
        
        for key, value in kwargs.items():
            if hasattr(self, key) and hasattr(getattr(self, key), '__len__'):
                # This is likely a data column that can be used for filtering
                column_filters[key] = value
            else:
                # This is a plot styling parameter
                plot_kwargs[key] = value
        
        # Build combined mask from mask parameter and column filters
        if mask is None:
            mask = np.ones(len(self[x]), dtype=bool)
        else:
            mask = mask.copy()  # Don't modify the original mask
        
        # Apply column filtering
        filter_info = []
        for col_name, filter_value in column_filters.items():
            try:
                col_data = getattr(self, col_name)
                if hasattr(col_data, '__len__') and len(col_data) == len(mask):
                    col_mask = col_data == filter_value
                    mask = mask & col_mask
                    filter_info.append(f"{col_name}={filter_value}")
                else:
                    warnings.warn(f"Column '{col_name}' exists but has incompatible shape for filtering")
            except AttributeError:
                warnings.warn(f"Column filtering requested ({col_name}={filter_value}) but '{col_name}' column not found in data")
        
        # Handle sign flipping with '-' prefix
        flip_left = False
        flip_right = False
        y_left_column = y_left
        y_right_column = y_right
        
        if y_left.startswith('-'):
            flip_left = True
            y_left_column = y_left[1:]  # Remove the '-' prefix
        if y_right.startswith('-'):
            flip_right = True
            y_right_column = y_right[1:]  # Remove the '-' prefix
        
        # Get data and apply sign flipping if requested
        x_data = self[x][mask]
        y_left_data = self[y_left_column][mask]
        y_right_data = self[y_right_column][mask]
        
        if flip_left:
            y_left_data = -y_left_data
        if flip_right:
            y_right_data = -y_right_data
        
        # Plot data
        if hue:
            # Plot with hue grouping
            for val in np.unique(self[hue][mask]):
                hue_mask = self[hue][mask] == val
                ax_left.plot(x_data[hue_mask], y_left_data[hue_mask], 
                           color=color_left, label=f'{y_left} ({val})', **plot_kwargs)
                ax_right.plot(x_data[hue_mask], y_right_data[hue_mask], 
                            color=color_right, label=f'{y_right} ({val})', **plot_kwargs)
        else:
            # Use numpy masked arrays to hide unwanted values while preserving structure
            x_plot = np.ma.array(self[x], mask=~mask)
            y_left_plot = np.ma.array(self[y_left_column], mask=~mask)
            y_right_plot = np.ma.array(self[y_right_column], mask=~mask)
            
            if flip_left:
                y_left_plot = np.ma.multiply(y_left_plot, -1)
            if flip_right:
                y_right_plot = np.ma.multiply(y_right_plot, -1)
            
            # Suppress the masked array conversion warning during plotting
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan')
                ax_left.plot(x_plot, y_left_plot, color=color_left, 
                            label=self.data_columns.get(y_left, y_left), **plot_kwargs)
                ax_right.plot(x_plot, y_right_plot, color=color_right, 
                             label=self.data_columns.get(y_right, y_right), **plot_kwargs)
        
        # Apply advanced scaling if requested
        if scale_range is not None:
            self._apply_matched_scaling(ax_left, ax_right, y_left_data, y_right_data,
                                      percentile_range, scale_range, tick_spacing)
        
        # Set axis labels and colors (with sign flip indicators)
        left_label = self.data_columns.get(y_left_column, y_left_column)
        right_label = self.data_columns.get(y_right_column, y_right_column)
        
        if flip_left:
            left_label = f"- {left_label}"
        if flip_right:
            right_label = f"- {right_label}"
        
        ax_left.set_xlabel(self.data_columns.get(x, x))
        ax_left.set_ylabel(left_label, color=color_left)
        ax_right.set_ylabel(right_label, color=color_right)
        
        # Color the axis ticks and spines
        ax_left.tick_params(axis='y', labelcolor=color_left)
        ax_right.tick_params(axis='y', labelcolor=color_right)
        ax_left.spines['left'].set_color(color_left)
        ax_right.spines['right'].set_color(color_right)
        
        # Apply additional axis formatting
        if ax_left_kws:
            ax_left.set(**ax_left_kws)
        if ax_right_kws:
            ax_right.set(**ax_right_kws)
        
        # Create combined legend
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_left.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Set title
        title = f'{self.fname}: {self.data_columns.get(y_left, y_left)} vs {self.data_columns.get(y_right, y_right)}'
        if filter_info:
            title += f' ({", ".join(filter_info)})'
        if scale_range is not None:
            if scale_range == 'auto':
                scale_info = "auto"
            elif isinstance(scale_range, (int, float)):
                scale_info = f"fixed ({scale_range:.3f})"
            else:
                scale_info = str(scale_range)
            title += f' (matched scale: {scale_info})'
        ax_left.set_title(title)
        
        plt.tight_layout()
        return fig, (ax_left, ax_right)
    
    def _apply_matched_scaling(self, ax_left, ax_right, y_left_data, y_right_data,
                             percentile_range, scale_range, tick_spacing):
        """
        Apply matched scaling logic for dual y-axis plots.
        
        This is useful for plotting similar quantities (e.g., dual potentials)
        where you want the same scale range for visual comparison.
        
        Args:
            scale_range: 'auto' for auto-calculation or float for fixed range
        """
        # Calculate data ranges using percentiles to exclude outliers
        left_min = np.percentile(y_left_data, percentile_range[0])
        left_max = np.percentile(y_left_data, percentile_range[1])
        right_min = np.percentile(y_right_data, percentile_range[0])
        right_max = np.percentile(y_right_data, percentile_range[1])
        
        # Determine scale range
        if scale_range == 'auto' or scale_range is None:
            # Auto-calculate from data
            left_range = left_max - left_min
            right_range = right_max - right_min
            scale_range = max(left_range, right_range) * 1.2  # Add 20% padding
        # else: use the provided numeric scale_range value
        
        # Function to round to tick spacing boundaries
        def round_to_spacing(value, spacing):
            return np.round(value / spacing) * spacing
        
        # Calculate centers and round them to tick spacing boundaries
        left_center = round_to_spacing((left_min + left_max) / 2, tick_spacing)
        right_center = round_to_spacing((right_min + right_max) / 2, tick_spacing)
        
        # Calculate limits from rounded centers
        left_lim_min = left_center - scale_range/2
        left_lim_max = left_center + scale_range/2
        right_lim_min = right_center - scale_range/2
        right_lim_max = right_center + scale_range/2
        
        # Set the same scale (range) for both axes
        ax_left.set_ylim(left_lim_min, left_lim_max)
        ax_right.set_ylim(right_lim_min, right_lim_max)
        
        # Create tick positions at specified intervals
        left_ticks = np.arange(
            np.ceil(left_lim_min / tick_spacing) * tick_spacing,
            left_lim_max + tick_spacing/2,
            tick_spacing
        )
        right_ticks = np.arange(
            np.ceil(right_lim_min / tick_spacing) * tick_spacing,
            right_lim_max + tick_spacing/2,
            tick_spacing
        )
        
        ax_left.set_yticks(left_ticks)
        ax_right.set_yticks(right_ticks)
        
        # Format tick labels with appropriate precision
        decimal_places = max(0, int(-np.log10(tick_spacing)) + 1)
        formatter = plt.FuncFormatter(lambda x, p: f'{x:.{decimal_places}f}')
        ax_left.yaxis.set_major_formatter(formatter)
        ax_right.yaxis.set_major_formatter(formatter)

    def plotyy(self, **kwargs):
        """
        Deprecated: Use plot_dual_y() instead.
        
        This method is maintained for backward compatibility but plot_dual_y()
        offers more features and better scaling options.
        """
        warnings.warn(
            "plotyy() is deprecated. Use plot_dual_y() instead for better features and scaling options.",
            DeprecationWarning,
            stacklevel=2
        )
        # Handle legacy use_matched_scaling parameter
        if 'use_matched_scaling' in kwargs:
            if kwargs.pop('use_matched_scaling'):
                kwargs['scale_range'] = 'auto'
        
        # Remove advanced scaling options that don't exist in old method
        old_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['percentile_range', 'tick_spacing']}
        return self.plot_dual_y(**old_kwargs)

    @optional_return_figure
    def plot_overview(self, figsize=(15, 10), max_cols=3):
        """
        Create a grid plot overview of all available data columns.
        
        Args:
            figsize: Figure size as (width, height) tuple
            max_cols: Maximum number of columns in the grid
            return_figure: If True, return (fig, axes). If False, show plot and return None.
        """
        # Get all available data columns (exclude timestamp for plotting)
        plot_columns = [col for col in self.data_columns.keys() 
                       if col != 'timestamp' and hasattr(self, col) and len(getattr(self, col)) > 0]
        
        if not plot_columns:
            print("No data columns available for plotting")
            return None, None
            
        # Calculate grid dimensions
        n_plots = len(plot_columns)
        n_cols = min(max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Use timestamp as x-axis data
        if hasattr(self, 'timestamp') and len(self.timestamp) > 0:
            x_data = self.timestamp
        else:
            # Fallback to array indices
            first_col = getattr(self, plot_columns[0])
            x_data = np.arange(len(first_col))
        
        plot_count = 0
        for i, col in enumerate(plot_columns):
            try:
                # Get the data and check if it's numeric
                y_data = getattr(self, col)
                
                # Skip non-numeric columns
                if not np.issubdtype(y_data.dtype, np.number):
                    print(f"Skipping non-numeric column '{col}' (dtype: {y_data.dtype})")
                    continue
                
                row, col_idx = divmod(plot_count, n_cols)
                ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                
                # Plot the data directly - matplotlib handles masked arrays natively
                # Suppress the specific warning about masked element conversion during plotting
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan')
                    ax.plot(x_data, y_data, 'b-', linewidth=1, alpha=0.8)
                
                # Set title showing both column name and display name
                display_name = self.data_columns.get(col, col)
                title = f"{col} ({display_name})" if col != display_name else col
                ax.set_title(title, fontsize=10)
                ax.set_ylabel(display_name)
                ax.grid(True, alpha=0.3)
                
                # Remove x-axis labels as requested
                ax.set_xticklabels([])
                
                # Format y-axis for better readability
                if len(y_data) > 0:
                    y_range = np.ptp(y_data)
                    if y_range < 1e-6:
                        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
                
                plot_count += 1
                
            except Exception as e:
                print(f"Error plotting column '{col}': {e}")
                print(f"  Column dtype: {getattr(self, col).dtype if hasattr(self, col) else 'N/A'}")
                print(f"  Column shape: {getattr(self, col).shape if hasattr(self, col) else 'N/A'}")
                continue
        
        # Hide unused subplots
        for i in range(plot_count, n_rows * n_cols):
            row, col_idx = divmod(i, n_cols)
            ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
            ax.set_visible(False)
        
        # Add overall title
        fig.suptitle(f"Data Overview: {self.fname}", fontsize=14, fontweight='bold')
        return fig, axes

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

