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
import json
from datetime import datetime
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
                aux.load_data()
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
    continuous_data: bool = False # False for discrete data only (no interpolation, no data columns)
    has_visualization: bool = True  # Override to False for text-only sources
    name: str = ""  # Override with a unique identifier for this data source

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        '''Initialize the auxiliary data source with auxiliary folders.
        
        Args:
            auxiliary_folders: List of (folder_name, folder_path) tuples
        '''
        self.auxiliary_folders = auxiliary_folders

    # --- Data import methods ---
    @abstractmethod
    def load_data(self) -> Optional['AuxiliaryDataSource']:
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

    def interpolate_data_columns(self, main_timestamp, main_potential = None) -> dict[str, np.ndarray]:
        '''Interpolate auxiliary data columns to align with the main time series.

        Args:
            main_timestamp: The timestamp array of the main data
            main_potential: The potential array of the main data

        Returns:
            A dictionary mapping auxiliary column names to their interpolated values
        '''
        out = {}
        for column_name in self.main_data_columns:
            interp_data_column = self.interpolate_column_robust(column_name, main_timestamp)
            out[column_name] = interp_data_column
        return out

    def interpolate_column_robust(self, column_name: str, main_timestamp: np.ndarray) -> np.ndarray:
        '''Robust interpolation handling all overlap scenarios.
        
        Handles three cases:
        1. Full overlap: aux data covers entire main timespan
        2. Partial overlap: aux data covers part of main timespan (NaN for non-overlapping)
        3. No overlap: returns NaN array

        Args:
            column_name: The name of the auxiliary column to interpolate
            main_timestamp: The timestamp array of the main data
        
        Returns:
            Interpolated data array with same length as main_timestamp
        '''
        aux_data = getattr(self, column_name, None)
        aux_timestamps = getattr(self, 'timestamp', None)
        
        # Case 3: No aux data available
        if aux_data is None or aux_timestamps is None:
            logger.debug(f"No {column_name} data available, returning NaN array")
            return np.full(len(main_timestamp), np.nan)
        
        # Convert to numeric timestamps with timezone handling
        try:
            main_ts = pd.to_datetime(main_timestamp, utc=True).tz_convert(None)
            aux_ts = pd.to_datetime(aux_timestamps, utc=True).tz_convert(None)
            
            main_numeric = main_ts.astype('int64')
            aux_numeric = aux_ts.astype('int64')
        except Exception as e:
            logger.warning(f"Timestamp conversion failed for {column_name}: {e}")
            return np.full(len(main_timestamp), np.nan)
        
        # Check for any overlap
        main_start, main_end = main_numeric.min(), main_numeric.max()
        aux_start, aux_end = aux_numeric.min(), aux_numeric.max()
        
        has_overlap = not (main_end < aux_start or main_start > aux_end)
        
        if not has_overlap:
            # Case 3: No overlap
            logger.debug(f"No temporal overlap for {column_name}")
            return np.full(len(main_timestamp), np.nan)
        
        # Cases 1 & 2: Full or partial overlap
        # np.interp automatically handles partial overlap with NaN padding
        interpolated = np.interp(
            main_numeric,
            aux_numeric,
            aux_data,
            left=np.nan,   # NaN before aux data starts
            right=np.nan   # NaN after aux data ends
        )
        
        # Log interpolation statistics
        valid_points = np.sum(~np.isnan(interpolated))
        total_points = len(interpolated)
        overlap_pct = (valid_points / total_points) * 100 if total_points > 0 else 0
        
        logger.debug(f"{column_name}: {valid_points}/{total_points} points interpolated ({overlap_pct:.1f}% coverage)")
        
        return interpolated

    def interpolate_column(self, column_name: str, main_timestamp: np.ndarray) -> np.ndarray:
        '''Legacy method - calls robust interpolation.
        
        Args:
            column_name: The name of the auxiliary column to interpolate
            main_timestamp: The timestamp array of the main data
        '''
        return self.interpolate_column_robust(column_name, main_timestamp)

# Temporary place for bcs aux classes, not to be part of a generalized ectools

class PicoLogger(AuxiliaryDataSource):
    '''Auxiliary data source for PicoLogger data.'''

    timestamp: np.ndarray  # Timestamp array
    cell_potential: np.ndarray  # Cell potential in Volts

    name = "picologger"  # Unique identifier for this data source
    data_columns = {  # All data columns to be imported/calculated and stored
        'timestamp': 'Timestamp',
        'cell_pot': 'Cell Potential (V)',
    }
    main_data_columns = {
        'cell_pot': 'Cell Potential (V)',
        'counter_pot': 'Counter Potential (V)', # calculated column
    }
    has_visualization = True  # This source supports graphical plotting
    continuous_data = True  # Continuous data, supports interpolation

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        super().__init__(auxiliary_folders)

    def load_data(self) -> None:
        '''Load data from the PicoLogger auxiliary source.
        '''
        pico_files = []
        for folder_name, folder_path in self.auxiliary_folders:
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
    
    def interpolate_data_columns(self, main_timestamp, main_potential=None) -> dict[str, np.ndarray]:
        '''Handle PicoLogger's calculated columns with robust interpolation.
        
        Args:
            main_timestamp: The timestamp array of the main data
            main_potential: The potential array of the main data (required for counter_pot calculation)
        
        Returns:
            Dictionary with interpolated cell_pot and calculated counter_pot
        '''
        # Interpolate cell potential using robust method
        cell_pot = self.interpolate_column_robust('cell_pot', main_timestamp)
        
        # Calculate counter potential (only where cell_pot is valid)
        if main_potential is not None:
            counter_pot = np.where(
                np.isnan(cell_pot), 
                np.nan, 
                cell_pot - main_potential
            )
        else:
            logger.warning("PicoLogger: main_potential not provided, counter_pot will be NaN")
            counter_pot = np.full(len(main_timestamp), np.nan)

        return {
            'cell_pot': cell_pot,
            'counter_pot': counter_pot
        }


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
        'cascade_setpoint': 'Thermocouple Setpoint (°C)',
        'main_setpoint': 'Heating element Setpoint (°C)',
        'cascade_rate': 'Thermocouple Rate (°C/min)',
        'main_rate': 'Heating element Rate (°C/min)'
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
    continuous_data = True  # Continuous data, supports interpolation

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        super().__init__(auxiliary_folders)

    def load_data(self) -> None:
        '''Load data from the FurnaceLogger auxiliary source.
        '''
        data_temp = {
            'cascade': [],
            'main': [],
        }
        try:
            for folder_name, folder_path in self.auxiliary_folders:
                cascade_path = os.path.join(folder_path, 'CascadeController')
                if os.path.exists(cascade_path):
                    csv_files = [file for file in os.listdir(cascade_path) if file.endswith('.csv')]
                    for csv_file in csv_files:
                        cascade_data = pd.read_csv(os.path.join(cascade_path, csv_file), header=1)
                        cascade_data['timestamp'] = pd.to_datetime(
                            cascade_data['Date'] + ' ' + cascade_data['Time'], errors='coerce')
                        data_temp['cascade'].append(cascade_data)
                        logger.debug(f"FurnaceCascade: Loaded {len(cascade_data)} rows from {csv_file}")

            for folder_name, folder_path in self.auxiliary_folders:
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
        
        # Create figure with dual y-axes (wider to accommodate external legend)
        p_furnace = figure(
            title="Furnace Data",
            x_axis_label='Time',
            x_axis_type='datetime',
            width=1000,  # Increased width for external legend
            height=400,
            y_axis_label="Temperature (°C)"
        )
        
        # Define colors for different lines
        temp_colors = ['blue', 'orange', 'purple', 'brown']
        rate_colors = ['green', 'red', 'cyan', 'magenta']
        # Create ColumnDataSource
        source = ColumnDataSource(data=plot_data)
        
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
        
        # Create an invisible hover target that spans the full data range
        # This gives us a single clean tooltip instead of multiple overlapping ones
        hover_target = p_furnace.line(
            x='timestamp', y='cascade_temperature', source=source,
            alpha=0, line_width=0  # Completely invisible
        )
        
        # Create comprehensive tooltips for a single crosshair hover
        tooltips = [("Time", "@timestamp{%F %T}")]
        for col_name, display_name in self.data_columns.items():
            if col_name in plot_data and col_name != 'timestamp':
                if 'rate' in col_name:
                    tooltips.append((display_name, f"@{col_name}{{0.00}}"))
                else:
                    tooltips.append((display_name, f"@{col_name}{{0.0}}"))
        
        # Add hover tool ONLY to the invisible target
        hover = HoverTool(
            renderers=[hover_target],  # Only attach to our invisible line
            tooltips=tooltips,
            formatters={'@timestamp': 'datetime'},
            mode='vline'
        )
        p_furnace.add_tools(hover)
        
        # Create external legend using add_layout
        # Move the existing legend outside the plot area to the right
        p_furnace.add_layout(p_furnace.legend[0], 'right')
        
        # Add subtle grid for better readability
        p_furnace.grid.grid_line_alpha = 0.3
        
        show(p_furnace)


class JsonSource(AuxiliaryDataSource):
    '''Auxiliary data source for JSON metadata and settings.
    
    This source reads all JSON files in the auxiliary folders and can create
    specialized sub-sources like oxide sample collections.
    '''

    name = "json_source"  # Unique identifier for this data source
    has_visualization = False  # JSON source provides data, sub-sources handle visualization
    continuous_data = False  # JSON source is not continuous by default

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        # Initialize attributes
        self.oxide = None
        self.data = {}  # Simple dict to store all key-value pairs
        super().__init__(auxiliary_folders)

    def load_data(self) -> None:
        '''Load data from JSON files in auxiliary folders.
        '''
        json_data = {}
        json_files_found = []
        
        # Collect all JSON files from all auxiliary folders
        for folder_name, folder_path in self.auxiliary_folders:
            json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
            json_files_found.extend(json_files)
        
        if not json_files_found:
            logger.warning("JsonSource: No JSON files found in auxiliary folders")
            return
            
        # Load and merge all JSON data
        for json_file in json_files_found:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    json_data.update(file_data)
                    logger.debug(f"JsonSource: Loaded JSON from {json_file}")
            except Exception as e:
                logger.warning(f"JsonSource: Error loading {json_file}: {e}")
        
        # Store all data as key-value pairs
        self.data = json_data
        
        # Create oxide collection if oxide data exists
        if 'oxide' in json_data and 'salt_sampled' in json_data:
            try:
                self.oxide = Oxide.from_json_data(
                    json_data['oxide'], 
                    json_data['salt_sampled']
                )
                logger.info(f"JsonSource: Created oxide collection with {len(self.oxide.samples)} samples")
            except Exception as e:
                logger.warning(f"JsonSource: Failed to create oxide collection: {e}")
                self.oxide = None
        else:
            logger.debug("JsonSource: No oxide data found in JSON files")

    def visualize(self):
        '''Visualize JSON source data.
        
        For JsonSource, this displays a summary of loaded data and delegates
        visualization to sub-sources like oxide collection.
        '''
        print("=== JSON Source Data Summary ===")
        print(f"Data keys: {list(self.data.keys())}")
        
        if self.oxide:
            print(f"Oxide samples: {len(self.oxide.samples)}")
            self.oxide.visualize()
        else:
            print("Oxide samples: None")


class Oxide:
    '''Collection of oxide samples with analysis methods.
    
    This class manages multiple OxideSample objects and provides
    collection-level statistics and visualization.
    '''
    
    def __init__(self):
        self.samples: Dict[str, 'OxideSample'] = {}
    
    @classmethod
    def from_json_data(cls, oxide_data: Dict, salt_sampled_data: Dict) -> 'Oxide':
        '''Create Oxide collection from JSON data.
        
        Args:
            oxide_data: Dictionary with sample_id -> measurements mapping
            salt_sampled_data: Dictionary with sample_id -> timestamp mapping
            
        Returns:
            Oxide collection with populated samples
        '''
        collection = cls()
        
        for sample_id, measurements in oxide_data.items():
            try:
                if sample_id in salt_sampled_data:
                    timestamp = salt_sampled_data[sample_id]
                    sample = OxideSample(measurements, timestamp)
                    collection.add_sample(sample_id, sample)
                else:
                    logger.warning(f"No timestamp found for oxide sample {sample_id}")
            except Exception as e:
                logger.warning(f"Failed to create oxide sample {sample_id}: {e}")
        
        return collection
    
    def add_sample(self, sample_id: str, sample: 'OxideSample') -> None:
        '''Add a sample to the collection.'''
        self.samples[sample_id] = sample
    
    def get_sample(self, sample_id: str) -> Optional['OxideSample']:
        '''Get a sample by ID.'''
        return self.samples.get(sample_id)
    
    def mean_of_means(self) -> Optional[float]:
        '''Calculate mean of all sample means.'''
        if not self.samples:
            return None
        return np.mean([sample.mean for sample in self.samples.values()])
    
    def overall_stdev(self) -> Optional[float]:
        '''Calculate standard deviation across all sample means.'''
        if not self.samples:
            return None
        means = [sample.mean for sample in self.samples.values()]
        return np.std(means, ddof=1) if len(means) > 1 else 0.0
    
    def visualize(self) -> None:
        '''Display oxide sample information.'''
        if not self.samples:
            print("No oxide samples available")
            return
            
        print(f"\n=== Oxide Samples ({len(self.samples)} total) ===")
        print(f"Overall mean: {self.mean_of_means():.3f} ± {self.overall_stdev():.3f}")
        print("\nIndividual samples:")
        for sample_id, sample in self.samples.items():
            print(f"  {sample_id}: {sample}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples.values())
    
    def __getitem__(self, sample_id: str) -> 'OxideSample':
        return self.samples[sample_id]


class OxideSample:
    """
    Represents an oxide sample with multiple measurements and a corresponding timestamp.
    
    Attributes:
        measurements (List[float]): A list of individual measurement values.
        timestamp (datetime): The timestamp when the sample was taken.
    """
    
    def __init__(self, measurements: List[float], timestamp: str):
        """
        Initializes the OxideSample instance.
        
        Args:
            measurements (List[float]): A list of measurement values.
            timestamp (str): The timestamp as a string. It will be converted to a datetime object.
        
        Raises:
            ValueError: If measurements list is empty or contains non-numeric values.
            ValueError: If timestamp string is not in a recognizable datetime format.
        """
        if not measurements:
            raise ValueError("Measurements list cannot be empty.")
        if not all(isinstance(m, (int, float)) for m in measurements):
            raise ValueError("All measurements must be numeric values.")
        
        try:
            self.timestamp = datetime.fromisoformat(timestamp)
        except ValueError as ve:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from ve
        
        self.measurements = measurements
    
    @property
    def mean(self) -> float:
        """Calculates and returns the mean of the measurements."""
        return np.mean(self.measurements)
    
    @property
    def stdev(self) -> float:
        """Calculates and returns the standard deviation of the measurements."""
        return np.std(self.measurements, ddof=1)  # Sample standard deviation
    
    def __str__(self) -> str:
        """
        Returns a formatted string representation of the OxideSample.
        
        Format:
            "Oxide samples: mean ± stdev, sampled 'timestamp', ([measurements])"
        """
        return (f"Oxide sample: {self.mean:.3f} ± {self.stdev:.3f}, "
                f"sampled '{self.timestamp}', {self.measurements}")

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the OxideSample."""
        return (f"OxideSample(measurements={self.measurements}, "
                f"timestamp='{self.timestamp.isoformat()}')")
