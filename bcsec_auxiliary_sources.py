"""
Specialized auxiliary data sources for BCSEC electrochemistry experiments.

This module contains domain-specific auxiliary data sources that extend the
base ectools auxiliary framework for specialized experimental setups.

These classes should be moved to the bcsec repository for specialized use.
"""

import os
import logging
from typing import List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Import base framework from ectools
try:
    # When this file is in the bcsec repository
    import ectools
    from ectools.auxiliary_sources import AuxiliaryDataSource
except ImportError:
    # Fallback for local development in ectools workspace
    from auxiliary_sources import AuxiliaryDataSource

# Set up logging
logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available - plotting will be disabled")

try:
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d
    HAS_BOKEH = True
except ImportError:
    HAS_BOKEH = False
    logger.warning("Bokeh not available - interactive plotting will be disabled")


class PicoLogger(AuxiliaryDataSource):
    '''Auxiliary data source for PicoLogger potentiostat data.
    
    Loads CSV files from PicoLogger and provides cell potential measurements
    for auxiliary electrochemical analysis.
    '''
    
    # Type hints for data arrays
    timestamp: np.ndarray  # Timestamp array
    cell_pot: np.ndarray  # Cell potential in Volts

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
        
        Looks for CSV files containing 'pico' in their filename across all auxiliary folders.
        '''
        data_temp = []
        
        try:
            for folder_name, folder_path in self.auxiliary_folders:
                # Look for CSV files containing 'pico' in the filename (not in a subfolder)
                csv_files = [file for file in os.listdir(folder_path) 
                            if file.endswith('.csv') and 'pico' in file.lower()]
                
                for csv_file in csv_files:
                    file_path = os.path.join(folder_path, csv_file)
                    
                    # Try to read CSV with different header configurations
                    try:
                        pico_data = pd.read_csv(file_path, header=0)
                    except Exception:
                        pico_data = pd.read_csv(file_path, header=None)
                        
                    # Convert timestamp column
                    if 'Timestamp' in pico_data.columns:
                        pico_data['timestamp'] = pd.to_datetime(pico_data['Timestamp'], errors='coerce')
                    elif len(pico_data.columns) >= 2:
                        pico_data.columns = ['timestamp', 'cell_pot']
                        pico_data['timestamp'] = pd.to_datetime(pico_data['timestamp'], errors='coerce')
                    
                    data_temp.append(pico_data)
                    logger.debug(f"PicoLogger: Loaded {len(pico_data)} rows from {csv_file}")

            if not data_temp:
                logger.warning("PicoLogger: No data files found")
                return

            # Combine and sort data
            combined_data = pd.concat(data_temp, ignore_index=True).sort_values(by='timestamp')
            combined_data = combined_data.dropna(subset=['timestamp'])
            
            # Downsample if dataset is very large (>100k points)
            if len(combined_data) > 100000:
                step = len(combined_data) // 50000
                combined_data = combined_data.iloc[::step]
                logger.info(f"PicoLogger: Downsampled to {len(combined_data)} points for performance")
            
            # Store data as numpy arrays
            self.timestamp = combined_data['timestamp'].to_numpy()
            self.cell_pot = combined_data['cell_pot'].to_numpy()
            
            logger.info(f"PicoLogger: Successfully loaded {len(self.timestamp)} data points")
            
        except Exception as e:
            logger.error(f"PicoLogger: Error loading data: {e}")
            raise e

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
            # If no main potential provided, set counter_pot to NaN
            counter_pot = np.full_like(cell_pot, np.nan)
            logger.warning("PicoLogger: No main potential provided, counter_pot set to NaN")
        
        return {
            'cell_pot': cell_pot,
            'counter_pot': counter_pot
        }

    def visualize(self):
        '''Visualize the PicoLogger data using plots.'''
        self.plot()

    def plot_matplotlib(self, **kwargs: Any):
        '''Plot the auxiliary data using matplotlib.'''
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available for plotting")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        if hasattr(self, 'cell_pot') and len(self.cell_pot) > 0:
            ax.plot(self.timestamp, self.cell_pot, 'b-', linewidth=1.5, label='Cell Potential')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cell Potential (V)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        
        plt.title('PicoLogger - Cell Potential Data')
        plt.tight_layout()
        return fig, ax

    def plot_bokeh(self, **kwargs: Any):
        '''Plot the auxiliary data using Bokeh for interactive visualization.'''
        if not HAS_BOKEH:
            logger.error("Bokeh not available for interactive plotting")
            return self.plot_matplotlib(**kwargs)
        
        try:
            p = figure(
                width=800, height=400,
                title="PicoLogger - Cell Potential Data",
                x_axis_label='Time',
                y_axis_label='Cell Potential (V)',
                x_axis_type='datetime'
            )
            
            if hasattr(self, 'cell_pot') and len(self.cell_pot) > 0:
                # Create data source
                source = ColumnDataSource(data={
                    'timestamp': self.timestamp,
                    'cell_pot': self.cell_pot
                })
                
                # Add line plot
                p.line('timestamp', 'cell_pot', source=source, 
                      line_width=2, color='blue', alpha=0.8, legend_label='Cell Potential')
                
                # Add hover tool
                hover = HoverTool(tooltips=[
                    ('Time', '@timestamp{%F %T}'),
                    ('Cell Potential', '@cell_pot{0.000} V')
                ], formatters={'@timestamp': 'datetime'})
                p.add_tools(hover)
                
                # Position legend outside plot area
                p.legend.location = "top_left"
                p.legend.click_policy = "hide"
            
            show(p)
            return p
            
        except Exception as e:
            logger.warning(f"Bokeh plotting failed: {e}, falling back to matplotlib")
            return self.plot_matplotlib(**kwargs)

    def plot(self, **kwargs: Any):
        '''Plot the auxiliary data with Bokeh or matplotlib fallback.'''
        if HAS_BOKEH:
            return self.plot_bokeh(**kwargs)
        else:
            return self.plot_matplotlib(**kwargs)


class FurnaceLogger(AuxiliaryDataSource):
    '''Auxiliary data source for furnace temperature logging data.
    
    Loads CSV files from furnace controllers and provides temperature
    measurements and heating rate calculations.
    '''
    
    # Type hints for data arrays
    timestamp: np.ndarray  # Timestamp array
    cascade_temperature: np.ndarray  # Cascade thermocouple temperature
    main_temperature: np.ndarray  # Main heating element temperature
    cascade_rate: np.ndarray  # Cascade heating element rate of change
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
    main_data_columns = { # Data columns to be interpolated and added to experiment object
        'cascade_temperature': 'Thermocouple (°C)',
        'cascade_rate': 'Thermocouple Rate (°C/min)'
    }
    column_mapping = { # Column data names and headers in the csv files
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
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available for plotting")
            return

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
        
        # Rate plots on right y-axis
        ax2 = ax1.twinx()
        rate_colors = ['red', 'green']
        rate_handles = []
        
        for i, col_name in enumerate(rate_columns):
            if hasattr(self, col_name) and col_name in self.data_columns:
                display_name = self.data_columns[col_name]
                data = getattr(self, col_name)
                color = rate_colors[i % len(rate_colors)]
                
                line = ax2.plot(self.timestamp, data, label=display_name, 
                               color=color, linestyle=':', linewidth=1.5, alpha=0.7)
                rate_handles.extend(line)
        
        ax2.set_ylabel('Rate (°C/min)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        all_handles = temp_handles + rate_handles
        all_labels = [h.get_label() for h in all_handles]
        ax1.legend(all_handles, all_labels, loc='upper left')
        
        ax1.grid(True, alpha=0.3)
        plt.title('FurnaceLogger - Temperature and Rate Data')
        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_bokeh(self, **kwargs: Any):
        '''Plot the auxiliary data using Bokeh for interactive visualization.'''
        if not HAS_BOKEH:
            logger.error("Bokeh not available for interactive plotting")
            return self.plot_matplotlib(**kwargs)
        
        try:
            p = figure(
                width=900, height=500,
                title="FurnaceLogger - Temperature and Rate Data",
                x_axis_label='Time',
                y_axis_label='Temperature (°C)',
                x_axis_type='datetime'
            )
            
            if hasattr(self, 'cascade_temperature') and len(self.cascade_temperature) > 0:
                # Temperature data
                temp_source = ColumnDataSource(data={
                    'timestamp': self.timestamp,
                    'cascade_temp': self.cascade_temperature,
                    'main_temp': self.main_temperature,
                    'cascade_setpoint': getattr(self, 'cascade_setpoint', np.full_like(self.cascade_temperature, np.nan)),
                    'main_setpoint': getattr(self, 'main_setpoint', np.full_like(self.main_temperature, np.nan))
                })
                
                # Add temperature lines
                p.line('timestamp', 'cascade_temp', source=temp_source, 
                      line_width=2, color='blue', legend_label='Cascade Temperature')
                p.line('timestamp', 'main_temp', source=temp_source, 
                      line_width=2, color='orange', legend_label='Main Temperature')
                p.line('timestamp', 'cascade_setpoint', source=temp_source, 
                      line_width=1, color='blue', line_dash='dashed', alpha=0.7, legend_label='Cascade Setpoint')
                p.line('timestamp', 'main_setpoint', source=temp_source, 
                      line_width=1, color='orange', line_dash='dashed', alpha=0.7, legend_label='Main Setpoint')
                
                # Add second y-axis for rates
                p.extra_y_ranges = {"rate": Range1d(start=-50, end=50)}
                p.add_layout(LinearAxis(y_range_name="rate", axis_label="Rate (°C/min)"), 'right')
                
                # Rate data
                rate_source = ColumnDataSource(data={
                    'timestamp': self.timestamp,
                    'cascade_rate': getattr(self, 'cascade_rate', np.zeros_like(self.cascade_temperature)),
                    'main_rate': getattr(self, 'main_rate', np.zeros_like(self.main_temperature))
                })
                
                # Add rate lines
                p.line('timestamp', 'cascade_rate', source=rate_source, y_range_name="rate",
                      line_width=1.5, color='red', line_dash='dotted', alpha=0.8, legend_label='Cascade Rate')
                p.line('timestamp', 'main_rate', source=rate_source, y_range_name="rate",
                      line_width=1.5, color='green', line_dash='dotted', alpha=0.8, legend_label='Main Rate')
                
                # Add hover tools
                temp_hover = HoverTool(tooltips=[
                    ('Time', '@timestamp{%F %T}'),
                    ('Cascade Temp', '@cascade_temp{0.0} °C'),
                    ('Main Temp', '@main_temp{0.0} °C'),
                    ('Cascade SP', '@cascade_setpoint{0.0} °C'),
                    ('Main SP', '@main_setpoint{0.0} °C')
                ], formatters={'@timestamp': 'datetime'})
                
                rate_hover = HoverTool(tooltips=[
                    ('Time', '@timestamp{%F %T}'),
                    ('Cascade Rate', '@cascade_rate{0.0} °C/min'),
                    ('Main Rate', '@main_rate{0.0} °C/min')
                ], formatters={'@timestamp': 'datetime'})
                
                p.add_tools(temp_hover, rate_hover)
                
                # Position legend outside plot area
                p.legend.location = "top_left"
                p.legend.click_policy = "hide"
            
            show(p)
            return p
            
        except Exception as e:
            logger.warning(f"Bokeh plotting failed: {e}, falling back to matplotlib")
            return self.plot_matplotlib(**kwargs)

    def plot(self, **kwargs: Any):
        '''Plot the auxiliary data with Bokeh or matplotlib fallback.'''
        if HAS_BOKEH:
            return self.plot_bokeh(**kwargs)
        else:
            return self.plot_matplotlib(**kwargs)


class JsonSource(AuxiliaryDataSource):
    '''Auxiliary data source for JSON metadata and configuration files.
    
    This source reads all JSON files in the auxiliary folders and can create
    specialized sub-sources like oxide sample collections.
    '''

    name = "json_source"  # Unique identifier for this data source
    has_visualization = False  # JSON source provides data, sub-sources handle visualization
    continuous_data = False  # JSON source is not continuous by default

    def __init__(self, auxiliary_folders: List[Tuple[str, str]]) -> None:
        # Initialize attributes
        self.oxide = None
        super().__init__(auxiliary_folders)

    def load_data(self) -> None:
        '''Load data from JSON files in auxiliary folders.'''
        try:
            for folder_name, folder_path in self.auxiliary_folders:
                json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
                
                for json_file in json_files:
                    file_path = os.path.join(folder_path, json_file)
                    
                    if 'oxide' in json_file.lower():
                        # Try to create Oxide collection (placeholder for specialized class)
                        try:
                            # This would normally import from bcsec
                            # from bcsec.sample_classes import Oxide
                            # self.oxide = Oxide(file_path)
                            logger.info(f"JsonSource: Found oxide file {json_file} (Oxide class not implemented)")
                        except ImportError:
                            logger.warning("JsonSource: Oxide class not available, skipping oxide data")
                    else:
                        logger.debug(f"JsonSource: Found JSON file {json_file}")
                        
        except Exception as e:
            logger.error(f"JsonSource: Error loading JSON data: {e}")
            raise e

    def visualize(self):
        '''JSON source visualization depends on sub-sources.'''
        if self.oxide is not None and hasattr(self.oxide, 'visualize'):
            self.oxide.visualize()
        else:
            logger.info("JsonSource: No visualization available (discrete data source)")


# =============================================================================
# Standard Auxiliary Source Collection
# =============================================================================

# Standard set of BCSEC auxiliary data sources for typical experiments
STANDARD_SOURCES = [FurnaceLogger, PicoLogger, JsonSource]


# =============================================================================
# BCSEC Helper Functions (to be moved to bcsec.helper_functions)
# =============================================================================

def mc_filename_parser(_, fname: str) -> dict:
    """Parse the filename and return attribute dictionary. Expects a format like:
    '240926_15E_MCL19_cswWE1_SCAN-HOLD_LSV_STRIP_CO2_750C.DTA'
        Extracts:
            id_number: int
            id_letter: (optional) str
            id: str
            id_full: str
            we_number: int
            temperature: int
            mcl_number: int
            co2_number: float"""
    import re
    
    out = {}
    value_id = re.search(r'\d{6}_([0-9]+)([A-Za-z]*?)_', fname)
    value_id_full = re.search(r'(\d{6}_[0-9]+[A-Za-z]*?)_', fname)
    value_we_number = re.search(r'WE(\d+)', fname)
    value_temperature = re.search(r'_(\d+)C\.DTA', fname)
    mcl_number = re.search(r'_MCL(\d+)', fname)
    co2_number = re.search(r'_(\d+)CO2', fname)

    if value_id:
        out['id_number'] = int(value_id.group(1))
        out['id_letter'] = str(value_id.group(2)).lower() if value_id.group(2) else ''
        out['id'] = str(out['id_number']) + out['id_letter']
    out['id_full'] = str(value_id_full.group(1)) if value_id_full else None
    out['we_number'] = int(value_we_number.group(1)) if value_we_number else None
    out['temperature'] = int(value_temperature.group(1)) if value_temperature else None
    out['mcl_number'] = int(mcl_number.group(1)) if mcl_number else None

    if co2_number:
        # because 03 is 0.3 and 1 is 1.0
        co2_str = co2_number.group(1)
        if co2_str.startswith('0') and len(co2_str) > 1:
            out['co2_number'] = float(f"0.{co2_str[1:]}")
        else:
            out['co2_number'] = float(co2_str)
    else:
        out['co2_number'] = None

    return out
