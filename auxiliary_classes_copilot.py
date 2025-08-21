"""
Modular auxiliary data classes for ectools

This module provides object-oriented classes for handling different types of auxiliary data
channels (furnace, pico, oxide samples) with built-in importing and plotting capabilities.
These classes are designed to be modular and specific to BCS usage patterns.
"""

import os
import glob
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings

import pandas as pd
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from .config import requires_bokeh, get_config, Plotter

logger = logging.getLogger(__name__)


class AuxiliaryDataChannel(ABC):
    """Abstract base class for auxiliary data channels"""
    
    def __init__(self, name: str):
        self.name = name
        self.data: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}
        self.units: Dict[str, str] = {}
        self._loaded = False
    
    @abstractmethod
    def load_data(self, file_path: str) -> None:
        """Load data from file(s)"""
        pass
    
    @abstractmethod
    def plot(self, **kwargs) -> None:
        """Plot the auxiliary data"""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded"""
        return self._loaded
    
    def get_data_dict(self) -> Dict[str, np.ndarray]:
        """Get all data as a dictionary"""
        return self.data.copy()


class FurnaceLogger(AuxiliaryDataChannel):
    """Class for handling furnace temperature data from cascade and main controllers"""
    
    def __init__(self):
        super().__init__("furnace")
        self.units = {
            'cascade_celsius': '°C',
            'main_celsius': '°C',
            'cascade_setpoint': '°C',
            'main_setpoint': '°C',
            'heating_rate': '°C/min',
            'heating_rate_main': '°C/min'
        }
    
    def load_data(self, auxiliary_path: str) -> None:
        """
        Load furnace data from cascade and main controller CSV files
        
        Args:
            auxiliary_path (str): Path to auxiliary data directory
        """
        if not os.path.exists(auxiliary_path):
            raise FileNotFoundError(f"Auxiliary path {auxiliary_path} does not exist")
        
        # Load cascade controller data
        cascade_path = os.path.join(auxiliary_path, 'CascadeController')
        if os.path.exists(cascade_path):
            try:
                csv_files = [f for f in os.listdir(cascade_path) if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]
                    cascade_data = pd.read_csv(os.path.join(cascade_path, csv_file), header=1)
                    self.data['cascade_timestamp'] = pd.to_datetime(
                        cascade_data['Date'] + ' ' + cascade_data['Time'], 
                        errors='coerce'
                    ).to_numpy()
                    self.data['cascade_celsius'] = cascade_data['Cascade_Controller_PV'].to_numpy()
                    self.data['cascade_setpoint'] = cascade_data['Cascade_Controller_Working_SP'].to_numpy()
                else:
                    logger.warning("No CSV files found in cascade controller directory")
            except Exception as e:
                raise RuntimeError(f'Error reading cascade controller data: {e}') from e
        
        # Load main controller data
        main_path = os.path.join(auxiliary_path, 'MainController')
        if os.path.exists(main_path):
            try:
                csv_files = [f for f in os.listdir(main_path) if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]
                    main_data = pd.read_csv(os.path.join(main_path, csv_file), header=1)
                    self.data['main_timestamp'] = pd.to_datetime(
                        main_data['Date'] + ' ' + main_data['Time'], 
                        errors='coerce'
                    ).to_numpy()
                    self.data['main_celsius'] = main_data['Main_Controller_PV'].to_numpy()
                    self.data['main_setpoint'] = main_data['Main_Controller_Working_SP'].to_numpy()
                else:
                    logger.warning("No CSV files found in main controller directory")
            except Exception as e:
                raise RuntimeError(f'Error reading main controller data: {e}') from e
        
        # Synchronize timestamps and compute heating rates
        self._synchronize_timestamps()
        self._compute_heating_rates()
        self._loaded = True
    
    def _synchronize_timestamps(self) -> None:
        """Synchronize cascade and main controller timestamps"""
        if 'cascade_timestamp' in self.data and 'main_timestamp' in self.data:
            # Trim arrays to match lengths
            while len(self.data['cascade_timestamp']) != len(self.data['main_timestamp']):
                if len(self.data['cascade_timestamp']) > len(self.data['main_timestamp']):
                    self.data['cascade_timestamp'] = self.data['cascade_timestamp'][:-1]
                    self.data['cascade_celsius'] = self.data['cascade_celsius'][:-1]
                    self.data['cascade_setpoint'] = self.data['cascade_setpoint'][:-1]
                else:
                    self.data['main_timestamp'] = self.data['main_timestamp'][:-1]
                    self.data['main_celsius'] = self.data['main_celsius'][:-1]
                    self.data['main_setpoint'] = self.data['main_setpoint'][:-1]
            
            # Check if timestamps match
            if not np.array_equal(self.data['cascade_timestamp'], self.data['main_timestamp']):
                logger.warning("Timestamp mismatch between cascade and main controllers")
                # Keep only cascade data
                self.data.pop('main_timestamp', None)
                self.data.pop('main_celsius', None)
                self.data.pop('main_setpoint', None)
                self.data['timestamp'] = self.data['cascade_timestamp']
            else:
                self.data['timestamp'] = self.data['cascade_timestamp']
                # Remove individual timestamps
                self.data.pop('cascade_timestamp', None)
                self.data.pop('main_timestamp', None)
        elif 'cascade_timestamp' in self.data:
            self.data['timestamp'] = self.data['cascade_timestamp']
            self.data.pop('cascade_timestamp', None)
        elif 'main_timestamp' in self.data:
            self.data['timestamp'] = self.data['main_timestamp']
            self.data.pop('main_timestamp', None)
    
    def _compute_heating_rates(self) -> None:
        """Compute heating rates for available temperature data"""
        if 'timestamp' not in self.data:
            return
        
        ts = self.data['timestamp']
        ts_sec = ts.astype("datetime64[s]").astype("int64")
        dt = np.diff(ts_sec) / 60.0  # Convert to minutes
        
        # Compute heating rate for cascade temperature
        if 'cascade_celsius' in self.data:
            cascade_temp = self.data['cascade_celsius']
            d_temp = np.diff(cascade_temp)
            self.data['heating_rate'] = np.concatenate(([np.nan], d_temp / dt))
        
        # Compute heating rate for main temperature
        if 'main_celsius' in self.data:
            main_temp = self.data['main_celsius']
            d_temp_main = np.diff(main_temp)
            self.data['heating_rate_main'] = np.concatenate(([np.nan], d_temp_main / dt))
    
    def plot(self, show_heating_rate: bool = True, **kwargs) -> None:
        """
        Plot furnace temperature data with optional heating rate
        
        Args:
            show_heating_rate (bool): Whether to show heating rate on secondary axis
            **kwargs: Additional plotting arguments
        """
        if not self.is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        if 'timestamp' not in self.data:
            raise RuntimeError("No timestamp data available for plotting")
        
        # Try Bokeh first, fall back to matplotlib
        if BOKEH_AVAILABLE and get_config('plotter') == Plotter.BOKEH:
            self._plot_bokeh(show_heating_rate, **kwargs)
        elif MATPLOTLIB_AVAILABLE:
            self._plot_matplotlib(show_heating_rate, **kwargs)
        else:
            raise RuntimeError("No plotting backend available")
    
    @requires_bokeh
    def _plot_bokeh(self, show_heating_rate: bool, **kwargs) -> None:
        """Plot using Bokeh"""
        source = ColumnDataSource(data=self.data)
        
        # Create figure
        p = figure(
            title="Furnace Temperature Data",
            x_axis_label='Time',
            x_axis_type='datetime',
            width=kwargs.get('width', 800),
            height=kwargs.get('height', 400)
        )
        
        # Plot temperature data
        if 'cascade_celsius' in self.data:
            p.line(x='timestamp', y='cascade_celsius', source=source,
                   legend_label='Thermocouple', line_width=2, color="blue", name='cascade_line')
        
        if 'main_celsius' in self.data:
            p.line(x='timestamp', y='main_celsius', source=source,
                   legend_label='Heating Element', line_width=2, color='orange', name='main_line')
        
        # Add heating rate on secondary axis if requested
        if show_heating_rate and 'heating_rate' in self.data:
            p.extra_y_ranges = {"heating_rate": Range1d(start=-20, end=40)}
            p.add_layout(LinearAxis(y_range_name="heating_rate", axis_label="Heating Rate (°C/min)"), 'right')
            p.y_range = Range1d(start=-400, end=800)
            
            p.line(x='timestamp', y='heating_rate', source=source,
                   legend_label="Heating Rate (Cascade)", line_width=2, color="green",
                   y_range_name="heating_rate", name="heating_rate_line")
            
            if 'heating_rate_main' in self.data:
                p.line(x='timestamp', y='heating_rate_main', source=source,
                       legend_label="Heating Rate (Main)", line_width=2, color="red",
                       y_range_name="heating_rate", name="heating_rate_main_line")
        
        # Add hover tools
        self._add_bokeh_hover_tools(p)
        
        # Set axis labels
        p.yaxis.axis_label = "Temperature (°C)"
        
        show(p)
    
    def _plot_matplotlib(self, show_heating_rate: bool, **kwargs) -> None:
        """Plot using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available")
        
        fig, ax1 = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Plot temperature data
        if 'cascade_celsius' in self.data:
            ax1.plot(self.data['timestamp'], self.data['cascade_celsius'], 
                    label='Thermocouple', color='blue', linewidth=2)
        
        if 'main_celsius' in self.data:
            ax1.plot(self.data['timestamp'], self.data['main_celsius'], 
                    label='Heating Element', color='orange', linewidth=2)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Furnace Temperature Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add heating rate on secondary axis if requested
        if show_heating_rate and 'heating_rate' in self.data:
            ax2 = ax1.twinx()
            
            if 'heating_rate' in self.data:
                ax2.plot(self.data['timestamp'], self.data['heating_rate'], 
                        label='Heating Rate (Cascade)', color='green', linewidth=2, alpha=0.7)
            
            if 'heating_rate_main' in self.data:
                ax2.plot(self.data['timestamp'], self.data['heating_rate_main'], 
                        label='Heating Rate (Main)', color='red', linewidth=2, alpha=0.7)
            
            ax2.set_ylabel('Heating Rate (°C/min)')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def _add_bokeh_hover_tools(self, plot) -> None:
        """Add hover tools to Bokeh plot"""
        # Temperature hover tools
        if 'cascade_celsius' in self.data:
            hover_cascade = HoverTool(
                renderers=[plot.select_one({'name': 'cascade_line'})],
                tooltips=[
                    ("Time", "@timestamp{%F %T}"),
                    ("Cascade (°C)", "@cascade_celsius")
                ],
                formatters={'@timestamp': 'datetime'},
                mode='vline'
            )
            plot.add_tools(hover_cascade)
        
        if 'main_celsius' in self.data:
            hover_main = HoverTool(
                renderers=[plot.select_one({'name': 'main_line'})],
                tooltips=[
                    ("Time", "@timestamp{%F %T}"),
                    ("Main (°C)", "@main_celsius")
                ],
                formatters={'@timestamp': 'datetime'},
                mode='vline'
            )
            plot.add_tools(hover_main)
        
        # Heating rate hover tools
        if 'heating_rate' in self.data:
            hover_rate = HoverTool(
                renderers=[plot.select_one({"name": "heating_rate_line"})],
                tooltips=[
                    ("Time", "@timestamp{%F %T}"),
                    ("Heating Rate (°C/min)", "@heating_rate{0.000}")
                ],
                formatters={"@timestamp": "datetime"},
                mode="vline"
            )
            plot.add_tools(hover_rate)
        
        if 'heating_rate_main' in self.data:
            hover_rate_main = HoverTool(
                renderers=[plot.select_one({"name": "heating_rate_main_line"})],
                tooltips=[
                    ("Time", "@timestamp{%F %T}"),
                    ("Heating Rate Main (°C/min)", "@heating_rate_main{0.000}")
                ],
                formatters={"@timestamp": "datetime"},
                mode="vline"
            )
            plot.add_tools(hover_rate_main)


class PicoLogger(AuxiliaryDataChannel):
    """Class for handling pico potentiostat auxiliary data"""
    
    def __init__(self):
        super().__init__("pico")
        self.units = {'pot': 'V'}
    
    def load_data(self, auxiliary_path: str) -> None:
        """
        Load pico data from CSV files
        
        Args:
            auxiliary_path (str): Path to auxiliary data directory
        """
        if not os.path.exists(auxiliary_path):
            raise FileNotFoundError(f"Auxiliary path {auxiliary_path} does not exist")
        
        try:
            pico_files = glob.glob(os.path.join(auxiliary_path, '**', '*pico*.csv'), recursive=True)
            if not pico_files:
                logger.warning("No pico files found")
                return
            
            # Concatenate all pico files
            pico_data = pd.concat([pd.read_csv(file, header=0) for file in pico_files])
            
            # Convert first column to datetime
            self.data['timestamp'] = pd.to_datetime(
                pico_data.iloc[:, 0], errors='coerce', unit='s'
            ).to_numpy()
            
            # Get potential data and convert units if necessary
            pot_data = pico_data.iloc[:, 1].to_numpy()
            pot_column_name = pico_data.columns[1]
            
            if 'mV' in pot_column_name:
                self.data['pot'] = pot_data / 1000  # Convert mV to V
            elif 'V' in pot_column_name:
                self.data['pot'] = pot_data
            else:
                raise ValueError(f"Cannot determine potential units from column '{pot_column_name}'")
            
            self._loaded = True
            
        except Exception as e:
            raise RuntimeError(f'Error reading pico data: {e}') from e
    
    def plot(self, **kwargs) -> None:
        """
        Plot pico potential data
        
        Args:
            **kwargs: Additional plotting arguments
        """
        if not self.is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        if 'timestamp' not in self.data or 'pot' not in self.data:
            raise RuntimeError("No data available for plotting")
        
        # Try Bokeh first, fall back to matplotlib
        if BOKEH_AVAILABLE and get_config('plotter') == Plotter.BOKEH:
            self._plot_bokeh(**kwargs)
        elif MATPLOTLIB_AVAILABLE:
            self._plot_matplotlib(**kwargs)
        else:
            raise RuntimeError("No plotting backend available")
    
    @requires_bokeh
    def _plot_bokeh(self, **kwargs) -> None:
        """Plot using Bokeh"""
        source = ColumnDataSource(data=self.data)
        
        p = figure(
            title="Pico Potential Data",
            x_axis_label='Time',
            x_axis_type='datetime',
            width=kwargs.get('width', 800),
            height=kwargs.get('height', 400)
        )
        
        p.line(x='timestamp', y='pot', source=source,
               legend_label='Cell Potential', line_width=2, color='green', name='pico_line')
        
        # Add hover tool
        hover = HoverTool(
            renderers=[p.select_one({'name': 'pico_line'})],
            tooltips=[
                ("Time", "@timestamp{%F %T}"),
                ("Potential (V)", "@pot{0.000}")
            ],
            formatters={'@timestamp': 'datetime'},
            mode='vline'
        )
        p.add_tools(hover)
        
        p.yaxis.axis_label = "Cell Potential (V)"
        show(p)
    
    def _plot_matplotlib(self, **kwargs) -> None:
        """Plot using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        ax.plot(self.data['timestamp'], self.data['pot'], 
                label='Cell Potential', color='green', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Cell Potential (V)')
        ax.set_title('Pico Potential Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class OxideSample:
    """
    Represents an oxide sample with multiple measurements and timestamp
    """
    
    def __init__(self, measurements: List[float], timestamp: str):
        """
        Initialize oxide sample
        
        Args:
            measurements (List[float]): List of measurement values
            timestamp (str): Timestamp string (ISO format)
        """
        if not measurements:
            raise ValueError("Measurements list cannot be empty")
        if not all(isinstance(m, (int, float)) for m in measurements):
            raise ValueError("All measurements must be numeric")
        
        try:
            self.timestamp = datetime.fromisoformat(timestamp)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from e
        
        self.measurements = measurements
    
    @property
    def mean(self) -> float:
        """Calculate mean of measurements"""
        return np.mean(self.measurements)
    
    @property
    def stdev(self) -> float:
        """Calculate standard deviation of measurements"""
        return np.std(self.measurements, ddof=1)
    
    def __str__(self) -> str:
        return (f"Oxide sample: {self.mean:.3f} ± {self.stdev:.3f}, "
                f"sampled '{self.timestamp}', {self.measurements}")
    
    def __repr__(self) -> str:
        return (f"OxideSample(measurements={self.measurements}, "
                f"timestamp='{self.timestamp.isoformat()}')")


class OxideData(AuxiliaryDataChannel):
    """Class for handling oxide sample data"""
    
    def __init__(self):
        super().__init__("oxide")
        self.samples: Dict[str, OxideSample] = {}
        self.units = {'oxide': 'wt%'}  # Assuming weight percent
    
    def load_data(self, auxiliary_path: str) -> None:
        """
        Load oxide sample data from JSON files
        
        Args:
            auxiliary_path (str): Path to auxiliary data directory
        """
        if not os.path.exists(auxiliary_path):
            raise FileNotFoundError(f"Auxiliary path {auxiliary_path} does not exist")
        
        try:
            json_files = glob.glob(os.path.join(auxiliary_path, '**', '*.json'), recursive=True)
            aux_data = {}
            
            for file in json_files:
                with open(file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    aux_data.update(json_data)
            
            # Process oxide samples if available
            if 'oxide' in aux_data and 'salt_sampled' in aux_data:
                oxide_measurements = aux_data['oxide']
                salt_sampled = aux_data['salt_sampled']
                
                for sample_id, measurements in oxide_measurements.items():
                    try:
                        sample = OxideSample(measurements, salt_sampled[sample_id])
                        self.samples[sample_id] = sample
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error processing oxide sample {sample_id}: {e}")
            
            self._loaded = True
            
        except Exception as e:
            raise RuntimeError(f'Error reading oxide data: {e}') from e
    
    def plot(self, **kwargs) -> None:
        """
        Plot oxide sample data as bar chart with error bars
        
        Args:
            **kwargs: Additional plotting arguments
        """
        if not self.is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        if not self.samples:
            logger.warning("No oxide samples to plot")
            return
        
        # Try Bokeh first, fall back to matplotlib
        if BOKEH_AVAILABLE and get_config('plotter') == Plotter.BOKEH:
            self._plot_bokeh(**kwargs)
        elif MATPLOTLIB_AVAILABLE:
            self._plot_matplotlib(**kwargs)
        else:
            raise RuntimeError("No plotting backend available")
    
    @requires_bokeh
    def _plot_bokeh(self, **kwargs) -> None:
        """Plot using Bokeh"""
        sample_ids = list(self.samples.keys())
        means = [sample.mean for sample in self.samples.values()]
        stdevs = [sample.stdev for sample in self.samples.values()]
        
        source = ColumnDataSource(data=dict(
            sample_ids=sample_ids,
            means=means,
            stdevs=stdevs,
            upper=[m + s for m, s in zip(means, stdevs)],
            lower=[m - s for m, s in zip(means, stdevs)]
        ))
        
        p = figure(
            x_range=sample_ids,
            title="Oxide Sample Measurements",
            width=kwargs.get('width', 600),
            height=kwargs.get('height', 400)
        )
        
        # Plot bars
        p.vbar(x='sample_ids', top='means', width=0.6, source=source,
               color='steelblue', alpha=0.8, name='bars')
        
        # Add error bars
        p.segment(x0='sample_ids', y0='lower', x1='sample_ids', y1='upper',
                  source=source, line_width=2, color='black')
        
        # Add hover tool
        hover = HoverTool(
            renderers=[p.select_one({'name': 'bars'})],
            tooltips=[
                ("Sample ID", "@sample_ids"),
                ("Mean", "@means{0.000}"),
                ("Std Dev", "@stdevs{0.000}")
            ],
            mode='vline'
        )
        p.add_tools(hover)
        
        p.xaxis.axis_label = "Sample ID"
        p.yaxis.axis_label = "Oxide Content (wt%)"
        p.xaxis.major_label_orientation = 45
        
        show(p)
    
    def _plot_matplotlib(self, **kwargs) -> None:
        """Plot using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available")
        
        sample_ids = list(self.samples.keys())
        means = [sample.mean for sample in self.samples.values()]
        stdevs = [sample.stdev for sample in self.samples.values()]
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        
        bars = ax.bar(sample_ids, means, yerr=stdevs, capsize=5, 
                     color='steelblue', alpha=0.8, error_kw={'color': 'black'})
        
        ax.set_xlabel('Sample ID')
        ax.set_ylabel('Oxide Content (wt%)')
        ax.set_title('Oxide Sample Measurements')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary(self) -> str:
        """Get a summary of oxide samples"""
        if not self.samples:
            return "No oxide samples available"
        
        summary = ["Oxide samples:"]
        for sample_id, sample in self.samples.items():
            summary.append(f"  {sample_id}: {sample.mean:.3f} ± {sample.stdev:.3f} wt%")
        
        return "\n".join(summary)


class AuxiliaryDataManager:
    """Manager class for coordinating multiple auxiliary data channels"""
    
    def __init__(self):
        self.furnace = FurnaceLogger()
        self.pico = PicoLogger()
        self.oxide = OxideData()
        self.metadata: Dict[str, Any] = {}
    
    def load_all_data(self, auxiliary_path: str) -> None:
        """
        Load all auxiliary data from the specified path
        
        Args:
            auxiliary_path (str): Path to auxiliary data directory
        """
        try:
            self.furnace.load_data(auxiliary_path)
            logger.info("Furnace data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load furnace data: {e}")
        
        try:
            self.pico.load_data(auxiliary_path)
            logger.info("Pico data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load pico data: {e}")
        
        try:
            self.oxide.load_data(auxiliary_path)
            logger.info("Oxide data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load oxide data: {e}")
        
        # Load general metadata from JSON files
        self._load_metadata(auxiliary_path)
    
    def _load_metadata(self, auxiliary_path: str) -> None:
        """Load general metadata from JSON files"""
        try:
            json_files = glob.glob(os.path.join(auxiliary_path, '**', '*.json'), recursive=True)
            for file in json_files:
                with open(file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # Extract non-oxide/non-salt_sampled data as metadata
                    for key, value in json_data.items():
                        if key not in ['oxide', 'salt_sampled']:
                            self.metadata[key] = value
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
    
    def plot_all(self, **kwargs) -> None:
        """Plot all loaded auxiliary data"""
        if self.furnace.is_loaded:
            self.furnace.plot(**kwargs)
        
        if self.pico.is_loaded:
            self.pico.plot(**kwargs)
        
        if self.oxide.is_loaded:
            self.oxide.plot(**kwargs)
    
    def get_summary(self) -> str:
        """Get a summary of all auxiliary data"""
        summary = []
        
        # Add metadata summary
        if self.metadata:
            summary.append("Run metadata:")
            for key, value in self.metadata.items():
                summary.append(f"  {key}: {value}")
        
        # Add oxide summary
        if self.oxide.is_loaded:
            summary.append(self.oxide.get_summary())
        
        return "\n".join(summary)
