'''
Framework for importing of data columns from auxiliary sources.
This module provides a base class for auxiliary data sources, which can be extended
to implement specific data import logic for different file formats or data types.

This framework (and its extentions) should do 2 main things:
1. Store and provide visualization of auxiliary data sources for the folder, as a property
of an EcList object.
2. Integrate with the individual data containers (files) to add auxiliary data columns
which are interpolated (if applicable) or sliced to fit the main time axis of the data container.

As the source files, formatting and structure will vary, the user is expected to
extend this framework to implement the specific logic for their data files.
'''

import logging
import os
import glob
from typing import Any, Dict, Optional, Union, List, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from .config import requires_bokeh, bokeh_conf, get_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass  # For forward references if needed later


class AuxiliaryDataHandler:
    '''Collection of auxiliary data sources.
    
    This class manages multiple auxiliary data sources, allowing for easy access. This 
    should normally be initialized after parsing the main data files into an EcList object,
    "aux", and handle the interface to the auxiliary data sources.
    '''
    auxiliary_folders : List[Tuple[str, str]]  # List of (folder_name, folder_path) tuples
    sources : List[str]

    def __init__(self, main_path: str, aux_data_classes: List['AuxiliaryDataSource'], aux_folder_id = None) -> None:
        '''Initialize the auxiliary data handler with a path and a list of auxiliary classes.'''
        self.main_path = main_path
        self.aux_data_classes = aux_data_classes
        self.aux_folder_id = aux_folder_id
        self.sources = []
        self._search_auxiliary_folders()

    def __getitem__(self, key: str) -> Optional['AuxiliaryDataSource']:
        '''Get an auxiliary data source by its name.'''
        for aux in self.aux_data_classes:
            if hasattr(aux, 'name') and aux.name == key:
                return aux
        return None

    def __iter__(self):
        '''Allow iteration over auxiliary classes.'''
        return iter(self.aux_data_classes)

    def import_auxiliary_data(self) -> None:
        '''Import auxiliary data from all sources.'''
        for Aux_cls in self.aux_data_classes:
            try:
                aux = Aux_cls(self.auxiliary_folders)
                setattr(self, Aux_cls.name, aux)
                self.sources.append(Aux_cls.name)
            except Exception as e:
                logger.warning("Failed loading auxiliary data from %s: %s", aux, e)
                setattr(self, Aux_cls.name, None)  # Set to None if loading fails

    def _search_auxiliary_folders(self) -> None:
        '''Search for auxiliary folders in the specified path.
        
        Finds folders containing the aux_folder_id (partial match, case-insensitive).
        Falls back to exact 'auxiliary' folder if no partial matches found.
        
        Sets self.auxiliary_folders to a list of (folder_name, folder_path) tuples.
        '''
        
        logger = logging.getLogger(__name__)
        
        # Use provided identifier or get from config, fallback to 'auxiliary'
        folder_id = self.aux_folder_id or get_config('auxiliary_folder_identifier') or 'auxiliary'
        
        # Find all folders containing the identifier (partial match, case-insensitive)
        matching_folders = []
        try:
            for item in os.listdir(self.main_path):
                item_path = os.path.join(self.main_path, item)
                if os.path.isdir(item_path) and folder_id.lower() in item.lower():
                    matching_folders.append((item, item_path))
        except (OSError, PermissionError):
            pass
        
        # Fallback to exact 'auxiliary' folder if no partial matches found
        if not matching_folders:
            auxiliary_path = os.path.join(self.main_path, 'auxiliary')
            if os.path.exists(auxiliary_path):
                matching_folders = [('auxiliary', auxiliary_path)]
            else:
                raise FileNotFoundError(
                    f"No auxiliary folders found matching '{folder_id}' or exact 'auxiliary' folder"
                )
        
        logger.debug('Found %d auxiliary folders: %s', 
                    len(matching_folders), [folder[0] for folder in matching_folders])
        
        # Store the matching folders
        self.auxiliary_folders = matching_folders

    def visualize(self) -> None:
        '''Visualize all auxiliary data sources.'''
        for aux_name in self.sources:
            aux = getattr(self, aux_name, None)
            if aux:
                aux.visualize()


class AuxiliaryDataSource(ABC):
    '''Abstract base class for auxiliary data sources.
    
    To create a new auxiliary data source:
    
    1. Set has_visualization = True/False depending on whether you need plotting
    2. Override load_data() to implement your data loading logic  
    3. Override visualize() to implement your visualization logic
    4. If has_visualization=True, you MUST override both plot_matplotlib() and plot_bokeh()
    5. If has_visualization=False, implement custom display logic in visualize()
    
    Example for graphical source:
        class MyGraphicalSource(AuxiliaryDataSource):
            has_visualization = True
            
            def load_data(self, path): 
                # Load your data
                pass
                
            def visualize(self): 
                self.plot()  # Use default plot routing
                
            def plot_matplotlib(self, **kwargs):
                # Your matplotlib implementation
                pass
                
            @requires_bokeh  # Optional decorator
            def plot_bokeh(self, **kwargs):
                # Your bokeh implementation  
                pass
    
    Example for text-only source:
        class MyTextSource(AuxiliaryDataSource):
            has_visualization = False
            
            def load_data(self, path):
                # Load your data
                pass
                
            def visualize(self):
                # Your custom text display logic
                print("My data:", self.data)
    '''

    # Define data columns with display name. All plottable data columns should have a 'timestamp'
    # column to align with the main time series, and produce a useful general plot.

    data_columns: Dict[str, Union[str, Any]] = {
        'column_name': 'Column Description (Unit)',
    }
    # Define data columns which will be interpolated to the main time axis, with display units.
    # These will be used added to the main data_columns
    main_data_columns: List[str] = [
        'column_name'
    ]

    # Class variables and constants
    has_visualization: bool = True  # Override to False for text-only sources
    name: str = ""  # Override with a unique identifier for this data source

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        '''Initialize the auxiliary data source with auxiliary folders.
        
        Args:
            auxiliary_folders: List of (folder_name, folder_path) tuples
        '''
        self.load_data(auxiliary_folders)

    # --- Data import methods ---
    @abstractmethod
    def load_data(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        '''Load data from the auxiliary source.
        
        Args:
            auxiliary_folders: List of (folder_name, folder_path) tuples
        '''
        pass

    # --- Visualization methods ---
    @abstractmethod
    def visualize(self):
        '''Visualize the data, either using plot or text'''
        # Either call plot() or build a different visualisation 
        pass

    def plot(self):
        '''Plot the auxiliary data with bokeh or matplotlib.'''
        if not self.has_visualization:
            raise NotImplementedError(f"{self.__class__.__name__} does not support graphical plotting")
            
        if bokeh_conf: # Should be None if bokeh is not available
            self.plot_bokeh()
        else:
            self.plot_matplotlib()

    def plot_matplotlib(self, **kwargs: Any):
        '''Plot the auxiliary data using matplotlib.
        
        NOTE: This method must be overridden in subclasses that have has_visualization=True.
        Implement your matplotlib plotting logic here.
        
        Args:
            **kwargs: Additional plotting arguments passed to matplotlib functions
            
        Raises:
            NotImplementedError: If called on a class with has_visualization=False
                               or if not implemented in a subclass with has_visualization=True
        '''
        if not self.has_visualization:
            raise NotImplementedError(f"{self.__class__.__name__} does not support matplotlib plotting")
        
        # If we reach here, the subclass should have implemented this method
        raise NotImplementedError(
            f"{self.__class__.__name__} has has_visualization=True but plot_matplotlib() "
            "is not implemented. Please override this method with your matplotlib plotting logic."
        )

    def plot_bokeh(self, **kwargs: Any) -> None:
        '''Plot the auxiliary data using Bokeh.
        
        NOTE: This method must be overridden in subclasses that have has_visualization=True.
        Implement your Bokeh plotting logic here. Use the @requires_bokeh decorator
        if your implementation needs Bokeh to be available.
        
        Args:
            **kwargs: Additional plotting arguments passed to Bokeh functions
            
        Raises:
            NotImplementedError: If called on a class with has_visualization=False
                               or if not implemented in a subclass with has_visualization=True
        '''
        if not self.has_visualization:
            raise NotImplementedError(f"{self.__class__.__name__} does not support bokeh plotting")
        
        # If we reach here, the subclass should have implemented this method
        raise NotImplementedError(
            f"{self.__class__.__name__} has has_visualization=True but plot_bokeh() "
            "is not implemented. Please override this method with your Bokeh plotting logic."
        )


    # --- Data access methods ---
    def __getitem__(self, key: str) -> Any:
        '''Access attributes like a dictionary.
        
        Allows accessing class attributes such as data_columns, name, etc.
        
        Args:
            key: The attribute name to access
            
        Returns:
            The value of the attribute
            
        Raises:
            KeyError: If the attribute doesn't exist
        '''
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        '''Set attributes like a dictionary.
        
        Args:
            key: The attribute name to set
            value: The value to set
        '''
        setattr(self, key, value)

    def keys(self):
        '''Return available attribute names (similar to dict.keys()).'''
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]

    def items(self):
        '''Return attribute name-value pairs (similar to dict.items()).'''
        return [(key, getattr(self, key)) for key in self.keys()]

    # --- Data handling methods ---

    def interpolate_data_columns(self, main_timestamp, main_potential):
        '''Interpolate auxiliary data columns to align with the main time series.
        
        This method takes the auxiliary data loaded by this source and interpolates
        or processes it to match the timestamps of the main electrochemical data.
        The interpolation method depends on the data type - continuous data uses
        linear interpolation, while discrete events may use nearest neighbor or
        boolean masking approaches.
        
        Args:
            main_timestamp (numpy.ndarray): Timestamp array from the main electrochemical
                data that auxiliary data should be aligned to. Typically in seconds
                from experiment start or unix timestamps.
        
        Returns:
            Dict[str, Tuple[str, numpy.ndarray]]: Dictionary mapping column names to 
                tuples containing:
                - display_name (str): Human-readable column name with units for display,
                  as defined in main_data_columns
                - data_column (numpy.ndarray): Interpolated data array aligned with 
                  main_timestamp, same length as main_timestamp
                  
        Example:
            >>> aux_source = MyAuxiliarySource(folders)
            >>> main_time = np.array([0, 10, 20, 30])  # seconds
            >>> result = aux_source.interpolate_data_columns(main_time)
            >>> result
            {
                'temperature': ('Temperature (°C)', array([25.0, 26.1, 27.2, 28.3])),
                'pressure': ('Pressure (bar)', array([1.0, 1.0, 1.1, 1.1]))
            }
            
        Note:
            This method must be implemented by subclasses. The returned data will
            be added to the main electrochemical data files as additional columns
            accessible through the data_columns dictionary.
        '''
        # Implement interpolation logic here
        pass

# Temporary place for bcs aux classes, not to be part of a generalized ectools

class PicoLogger(AuxiliaryDataSource):
    '''Auxiliary data source for PicoLogger data.'''

    timestamp: np.ndarray  # Timestamp array
    cell_potential: np.ndarray  # Cell potential in Volts

    name = "picologger"  # Unique identifier for this data source
    data_columns = {  # All data columns to be imported/calculated and stored
        'timestamp': 'Timestamp',
        'cell_potential': 'Cell Potential (V)',
    }
    main_data_columns = {
        'cell_potential': 'Cell Potential (V)',
        'counter_potential': 'Counter Potential (V)',
    }
    has_visualization = True  # This source supports graphical plotting

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        super().__init__(auxiliary_folders)

    def load_data(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        '''Load data from the PicoLogger auxiliary source.
        
        Args:
            auxiliary_folders: List of (folder_name, folder_path) tuples
        '''
        pico_files = []
        for folder_name, folder_path in auxiliary_folders:
            pico_files.extend(glob.glob(os.path.join(folder_path, '**', '*pico*.csv'), recursive=True))
        
        if not pico_files:
            logger.warning("PicoLogger: No pico files found")
            return
            
        # Read all DataFrames and sort by first timestamp
        dfs = []
        for file_path in pico_files:
            # Load CSV with header=0 to properly parse column names (matches old helper_functions.py)
            df = pd.read_csv(file_path, header=0)
            if not df.empty:
                # Extract first timestamp for sorting (first column, first row)
                first_timestamp = pd.to_datetime(df.iloc[0, 0], unit='s', errors='coerce')
                df['_first_timestamp'] = first_timestamp
                dfs.append(df)
        
        if not dfs:
            logger.warning("PicoLogger: No valid pico files found")
            return
        
        # Sort DataFrames by their first timestamp
        dfs.sort(key=lambda df: df['_first_timestamp'].iloc[0])
        
        # Remove the temporary sorting column and concatenate
        for df in dfs:
            df.drop('_first_timestamp', axis=1, inplace=True)
        
        pico_data = pd.concat(dfs, ignore_index=True)

        # Set attributes using to_numpy() for consistency
        self.timestamp = pd.to_datetime(pico_data.iloc[:, 0], unit='s', errors='coerce').to_numpy()
        
        # Check the actual column name for units (second column header)
        column_name = pico_data.columns[1]  # This will be "Channel 4 Ave. (V)"
        if 'mV' in column_name:
            self.cell_potential = pico_data.iloc[:,1].to_numpy() / 1000  # Convert mV to V
        elif 'V' in column_name:
            self.cell_potential = pico_data.iloc[:,1].to_numpy()
        else:
            raise ValueError(f"Unsupported unit in cell potential column: {column_name}")

        logger.info(f"PicoLogger: Loaded {len(pico_data)} rows from {len(pico_files)} files")

    def visualize(self):
        '''Visualize the PicoLogger data using plots.'''
        self.plot()

    def plot_matplotlib(self, **kwargs: Any):
        '''Plot the auxiliary data using matplotlib.'''
        fig, ax = plt.subplots()
        ax.plot(self.timestamp, self.cell_potential, label='Cell Potential (V)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
        plt.show()

    @requires_bokeh
    def plot_bokeh(self, **kwargs: Any) -> None:
        '''Plot the auxiliary data using Bokeh.'''
        from bokeh.plotting import figure, show, output_notebook
        from bokeh.models import ColumnDataSource, HoverTool
        
        output_notebook()
        
        # Check if we have the required data
        if not hasattr(self, 'timestamp') or not hasattr(self, 'cell_potential'):
            logger.error("PicoLogger: No data available for plotting")
            return
            
        # Downsample data for better performance if dataset is large
        max_points = kwargs.get('max_points', 5000)  # Default to 5000 points for good performance
        timestamps = self.timestamp
        potentials = self.cell_potential
        
        if len(timestamps) > max_points:
            # Use every nth point to downsample
            step = len(timestamps) // max_points
            timestamps = timestamps[::step]
            potentials = potentials[::step]
            logger.info(f"PicoLogger: Downsampled from {len(self.timestamp)} to {len(timestamps)} points for plotting")
            
        # Prepare data dictionary
        plot_data = {
            'timestamp': timestamps,
            'cell_potential': potentials
        }
        
        # Create ColumnDataSource
        source = ColumnDataSource(data=plot_data)
        
        # Create figure
        p_pico = figure(
            title="PicoLogger Data: Cell Potential vs Time",
            x_axis_label='Time',
            x_axis_type='datetime',
            y_axis_label='Cell Potential (V)',
            width=800,
            height=400,
            toolbar_location="above"
        )
        
        # Plot cell potential data
        line_renderer = p_pico.line(
            x='timestamp', y='cell_potential', source=source,
            legend_label='Cell Potential (V)', line_width=2, 
            color='blue', alpha=0.8
        )
        
        # Create tooltips
        tooltips = [
            ("Time", "@timestamp{%F %T}"),
            ("Cell Potential", "@cell_potential{0.000} V")
        ]
        
        # Add hover tool (only to line for better performance)
        hover = HoverTool(
            renderers=[line_renderer],
            tooltips=tooltips,
            formatters={'@timestamp': 'datetime'},
            mode='vline'
        )
        p_pico.add_tools(hover)
        
        # Configure legend and grid
        p_pico.legend.location = "top_left"
        p_pico.legend.click_policy = "hide"
        p_pico.grid.grid_line_alpha = 0.3
        
        show(p_pico)


class FurnaceLogger(AuxiliaryDataSource):
    '''Auxiliary data source for FurnaceLogger data.
    This is tailored to the standard output from Carbolite Gero furnace. In the case of multiple 
    files (restarted logging) these should be properly sorted by timestamp when loading.
    '''

    timestamp: np.ndarray  # Timestamp array
    cascade_temperature: np.ndarray  # Cascade thermocouple temperature
    main_temperature: np.ndarray  # Main heating element temperature
    cascade_rate: np.ndarray  # Cascade thermocouple rate of change
    main_rate: np.ndarray  # Main heating element rate of change
    cascade_setpoint: np.ndarray  # Cascade thermocouple setpoint
    main_setpoint: np.ndarray  # Main heating element setpoint

    name = "furnacelogger"  # Unique identifier for this data source
    data_columns = { # All data columns to be imported/calculated and stored
        'timestamp': 'Timestamp',
        'cascade_temperature': 'Thermocouple (°C)',
        'main_temperature': 'Heating element (°C)',
        'cascade_rate': 'Thermocouple Rate (°C/min)',
        'main_rate': 'Heating element Rate (°C/min)',
        'cascade_setpoint': 'Thermocouple Setpoint (°C)',
        'main_setpoint': 'Heating element Setpoint (°C)'
    }
    main_data_columns = ( # Data columns to be interpolated and added to experiment object
        'cascade_temperature'
    )
    column_mapping = {
        'timestamp': 'timestamp',
        'cascade_temperature': 'Cascade_Controller_PV',
        'main_temperature': 'Main_Controller_PV',
        'cascade_setpoint': 'Cascade_Controller_Working_SP',
        'main_setpoint': 'Main_Controller_Working_SP'
    }
    has_visualization = True  # This source supports graphical plotting

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        super().__init__(auxiliary_folders)

    def load_data(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        '''Load data from the FurnaceLogger auxiliary source.
        
        Args:
            auxiliary_folders: List of (folder_name, folder_path) tuples
        '''
        data_temp = {
            'cascade': [],
            'main': [],
        }
        try:
            for folder_name, folder_path in auxiliary_folders:
                cascade_path = os.path.join(folder_path, 'CascadeController')
                if os.path.exists(cascade_path):
                    csv_files = [file for file in os.listdir(cascade_path) if file.endswith('.csv')]
                    for csv_file in csv_files:
                        cascade_data = pd.read_csv(os.path.join(cascade_path, csv_file), header=1)
                        cascade_data['timestamp'] = pd.to_datetime(
                            cascade_data['Date'] + ' ' + cascade_data['Time'], errors='coerce')
                        data_temp['cascade'].append(cascade_data)
                        logger.debug(f"FurnaceCascade: Loaded {len(cascade_data)} rows from {csv_file}")

            for folder_name, folder_path in auxiliary_folders:
                main_path = os.path.join(folder_path, 'MainController')
                if os.path.exists(main_path):
                    csv_files = [file for file in os.listdir(main_path) if file.endswith('.csv')]
                    for csv_file in csv_files:
                        main_data = pd.read_csv(os.path.join(main_path, csv_file), header=1)
                        main_data['timestamp'] = pd.to_datetime(
                            main_data['Date'] + ' ' + main_data['Time'], errors='coerce')
                        data_temp['main'].append(main_data)
                        logger.debug(f"FurnaceMain: Loaded {len(main_data)} rows from {csv_file}")
            # Combine and sort data by timestamp
            cascade = pd.concat(data_temp['cascade'], ignore_index=True).sort_values(by='timestamp')
            main = pd.concat(data_temp['main'], ignore_index=True).sort_values(by='timestamp')
            
            # Merge on timestamp to handle different lengths - only keep matching timestamps
            merged_data = pd.merge(cascade, main, on='timestamp', suffixes=('_cascade', '_main'), how='inner')
            
            
            # Check if we have any data after merging
            if len(merged_data) == 0:
                logger.warning("FurnaceLogger: No matching timestamps between cascade and main data")
                return
        except Exception as e:
            logger.error(f"FurnaceLogger: Error loading data: {e}")
            raise e
        
        try:
            # Set data columns
            for col_name, header in self.column_mapping.items():
                if header in merged_data.columns:
                    setattr(self, col_name, merged_data[header].to_numpy())
                else:
                    logger.warning(f"FurnaceLogger: Column '{header}' not found in merged data")
            # Calculate rate columns (convert from per second to per minute)
            self.cascade_rate = np.gradient(self.cascade_temperature, edge_order=2) * 60  # °C/min
            self.main_rate = np.gradient(self.main_temperature, edge_order=2) * 60  # °C/min

            # Check that all data_columns are present
            for col_name in self.data_columns.keys():
                assert hasattr(self, col_name)

        except Exception as e:
            logger.error(f"FurnaceLogger: Error processing data: {e}")
            return

    def visualize(self):
        '''Visualize the FurnaceLogger data using plots.'''
        self.plot()

    def plot_matplotlib(self, **kwargs: Any):
        '''Plot the auxiliary data using matplotlib.'''

        temp_columns = ('cascade_temperature', 'main_temperature', 'cascade_setpoint', 'main_setpoint')
        rate_columns = ('cascade_rate', 'main_rate')

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        # Temperature plots on left y-axis
        temp_colors = ['blue', 'orange', 'purple', 'brown']
        temp_handles = []
        
        for i, col_name in enumerate(temp_columns):
            if hasattr(self, col_name) and col_name in self.data_columns:
                display_name = self.data_columns[col_name]
                data = getattr(self, col_name)
                color = temp_colors[i % len(temp_colors)]
                line_style = '--' if 'setpoint' in col_name else '-'
                
                line = ax1.plot(self.timestamp, data, label=display_name, 
                               color=color, linestyle=line_style, linewidth=2)
                temp_handles.extend(line)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # Rate plots on right y-axis
        ax2 = ax1.twinx()
        rate_colors = ['green', 'red']
        rate_handles = []
        
        for i, col_name in enumerate(rate_columns):
            if hasattr(self, col_name) and col_name in self.data_columns:
                display_name = self.data_columns[col_name]
                data = getattr(self, col_name)
                color = rate_colors[i % len(rate_colors)]
                
                line = ax2.plot(self.timestamp, data, label=display_name, 
                               color=color, linewidth=2, alpha=0.7)
                rate_handles.extend(line)
        
        ax2.set_ylabel('Rate (°C/min)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Combine legends from both axes
        all_handles = temp_handles + rate_handles
        all_labels = [h.get_label() for h in all_handles]
        ax1.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.title('Furnace Data: Temperature and Heating Rate')
        plt.tight_layout()
        plt.show()

    @requires_bokeh
    def plot_bokeh(self, **kwargs: Any) -> None:
        '''Plot the auxiliary data using Bokeh.'''
        from bokeh.plotting import figure, show, output_notebook
        from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d
        
        output_notebook()
        
        # Prepare data dictionary using all available data columns
        plot_data = {}
        temperature_columns = []
        rate_columns = []
        
        # Organize data by type and build plot_data dict
        for col_name, display_name in self.data_columns.items():
            if hasattr(self, col_name) and getattr(self, col_name) is not None:
                plot_data[col_name] = getattr(self, col_name)
                
                # Categorize columns for plotting strategy
                if 'temperature' in col_name or 'setpoint' in col_name:
                    temperature_columns.append((col_name, display_name))
                elif 'rate' in col_name:
                    rate_columns.append((col_name, display_name))
        
        # Check if we have timestamp data
        if 'timestamp' not in plot_data:
            logger.error("FurnaceLogger: No timestamp data available for plotting")
            return
            
        # Create ColumnDataSource
        source = ColumnDataSource(data=plot_data)
        
        # Create figure with dual y-axes (wider to accommodate side legend)
        p_furnace = figure(
            title="Furnace Data",
            x_axis_label='Time',
            x_axis_type='datetime',
            width=1000,  # Increased width for side legend
            height=400,
            y_axis_label="Temperature (°C)"
        )
        
        # Define colors for different lines
        temp_colors = ['blue', 'orange', 'purple', 'brown']
        rate_colors = ['green', 'red', 'cyan', 'magenta']
        
        # Plot temperature data on left y-axis
        temp_renderers = []
        for i, (col_name, display_name) in enumerate(temperature_columns):
            color = temp_colors[i % len(temp_colors)]
            line_style = 'dashed' if 'setpoint' in col_name else 'solid'
            renderer = p_furnace.line(
                x='timestamp', y=col_name, source=source,
                legend_label=display_name, line_width=2, 
                color=color, line_dash=line_style
            )
            temp_renderers.append(renderer)
        
        # Add extra y-axis for heating rate if we have rate data
        if rate_columns:
            p_furnace.extra_y_ranges = {"rate": Range1d(start=-20, end=60)}  # Adjusted for °C/min
            p_furnace.add_layout(LinearAxis(y_range_name="rate", axis_label="Rate (°C/min)"), 'right')
            
            # Plot rate data on right y-axis
            for i, (col_name, display_name) in enumerate(rate_columns):
                color = rate_colors[i % len(rate_colors)]
                p_furnace.line(
                    x='timestamp', y=col_name, source=source,
                    legend_label=display_name, line_width=2, 
                    color=color, y_range_name="rate"
                )
        
        # Set temperature y-axis range
        p_furnace.y_range = Range1d(start=0, end=800)
        
        # Create comprehensive tooltips for a single crosshair hover
        tooltips = [("Time", "@timestamp{%F %T}")]
        for col_name, display_name in self.data_columns.items():
            if col_name in plot_data and col_name != 'timestamp':
                if 'rate' in col_name:
                    tooltips.append((display_name, f"@{col_name}{{0.00}}"))
                else:
                    tooltips.append((display_name, f"@{col_name}{{0.0}}"))
        
        # Add single hover tool that shows all data at cursor position
        hover = HoverTool(
            tooltips=tooltips,
            formatters={'@timestamp': 'datetime'},
            mode='vline'  # Single vertical line shows all values
        )
        p_furnace.add_tools(hover)
        
        # Configure legend on the right side with more space
        p_furnace.legend.location = "center_right"
        p_furnace.legend.click_policy = "hide"
        p_furnace.legend.spacing = 10  # Add spacing between legend items
        p_furnace.legend.margin = 10   # Add margin around legend
        
        # Add subtle grid for better readability
        p_furnace.grid.grid_line_alpha = 0.3
        
        show(p_furnace)
