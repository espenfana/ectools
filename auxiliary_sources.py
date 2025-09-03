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
        '''Get an auxiliary data source by its name using dictionary syntax.'''
        return getattr(self, key, None)

    def __contains__(self, key: str) -> bool:
        '''Check if an auxiliary data source exists.'''
        return hasattr(self, key) and getattr(self, key) is not None

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

    def __contains__(self, key):
        '''Check if attribute exists (for use with 'in' operator)'''
        return hasattr(self, key)

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
        aux_timestamps = getattr(self, '_timestamp_int64', None)  # Use internal int64 storage
        if aux_timestamps is None:
            aux_timestamps = getattr(self, 'timestamp', None)  # Fallback to property getter
        
        # Case 3: No aux data available
        if aux_data is None or aux_timestamps is None:
            logger.debug(f"No {column_name} data available, returning NaN array")
            return np.full(len(main_timestamp), np.nan)
        
        # Convert to numeric timestamps with timezone handling
        try:
            # Main timestamps are datetime objects from ElectroChemistry, convert them
            main_numeric = pd.to_datetime(main_timestamp).astype('int64')
            
            # Aux timestamps are already int64 nanoseconds from optimized loading
            if isinstance(aux_timestamps[0] if len(aux_timestamps) > 0 else None, (int, np.integer)):
                aux_numeric = aux_timestamps  # Already converted
            else:
                aux_numeric = pd.to_datetime(aux_timestamps).astype('int64')
            
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

# NOTE: Specialized auxiliary data sources have been moved to separate files
# for domain-specific implementations. This file contains only the base framework.
#
# Example usage for specialized sources:
# from bcsec.auxiliary_sources import FurnaceLogger, PicoLogger, JsonSource
