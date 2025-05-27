"""
Optimized auxiliary data association methods for ectools

This module contains improved versions of auxiliary data processing methods
with better error handling, performance, and maintainability.
"""

import logging
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd


class AuxiliaryDataProcessor:
    """Handles association of auxiliary data with electrochemistry files."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def associate_auxiliary_data(self, aux: Dict[str, Any], ec_file) -> Dict[str, Dict]:
        """
        Associate auxiliary data with an electrochemistry file's time span.
        
        Args:
            aux: Dictionary containing auxiliary data (pico, furnace, etc.)
            ec_file: ElectroChemistry file object with timestamp data
            
        Returns:
            Dictionary with processed auxiliary data for this file
        """
        if not hasattr(ec_file, 'timestamp') or len(ec_file.timestamp) == 0:
            self.logger.warning(f'No timestamp data in file {ec_file.fname}')
            return {'pico': {}, 'furnace': {}}
        
        # Convert timestamps once
        tstart = np.datetime64(ec_file.timestamp[0])
        tend = np.datetime64(ec_file.timestamp[-1])
        
        result = {'pico': {}, 'furnace': {}}
        
        # Process each auxiliary data type
        if 'pico' in aux:
            result['pico'] = self._process_pico_data(aux['pico'], ec_file, tstart, tend)
            
        if 'furnace' in aux:
            result['furnace'] = self._process_furnace_data(aux['furnace'], ec_file, tstart, tend)
            
        return result
    
    def _process_pico_data(self, pico_data: Dict, ec_file, tstart, tend) -> Dict:
        """Process pico auxiliary data."""
        if not pico_data or 'timestamp' not in pico_data or pico_data['timestamp'] is None:
            self.logger.debug('No valid pico timestamp data available')
            return {}
        
        try:
            # Convert timestamps with timezone handling
            ts_file = pd.to_datetime(ec_file.timestamp, utc=True).tz_convert(None)
            pico_ts = pd.to_datetime(pico_data['timestamp'], utc=True).tz_convert(None)
            
            # Find overlapping time range
            overlap_start = max(ts_file.min(), pico_ts.min())
            overlap_end = min(ts_file.max(), pico_ts.max())
            
            if overlap_start >= overlap_end:
                self.logger.debug(f'No time overlap between pico and file {ec_file.fname}')
                return {}
            
            # Create mask for valid file timestamps
            valid_mask = (ts_file >= overlap_start) & (ts_file <= overlap_end)
            
            if not valid_mask.any():
                self.logger.debug(f'No matching pico data found for file {ec_file.fname}')
                return {}
            
            ts_overlap = ts_file[valid_mask]
            
            # Interpolate pico potential data
            interp_pot = np.interp(
                ts_overlap.astype('int64'),
                pico_ts.astype('int64'),
                pico_data['pot'],
                left=np.nan,
                right=np.nan
            )
            
            # Build result dictionary
            result = {
                'pot': interp_pot,
                'time': np.array(ec_file.time)[valid_mask],
                'timestamp': np.array(ec_file.timestamp)[valid_mask],
                'counter_pot': np.array(ec_file.pot)[valid_mask] - interp_pot
            }
            
            self.logger.debug(f'Pico data associated with file {ec_file.fname}, {len(interp_pot)} points')
            return result
            
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self.logger.error(f'Error processing pico data for {ec_file.fname}: {e}')
            return {}
    
    def _process_furnace_data(self, furnace_data: Dict, ec_file, tstart, tend) -> Dict:
        """Process furnace auxiliary data."""
        if not furnace_data or 'timestamp' not in furnace_data or furnace_data['timestamp'] is None:
            self.logger.debug('No valid furnace timestamp data available')
            return {}
        
        try:
            # Create time range mask
            furnace_mask = (furnace_data['timestamp'] >= tstart) & (furnace_data['timestamp'] <= tend)
            
            if not furnace_mask.any():
                self.logger.debug(f'No furnace data in time range for file {ec_file.fname}')
                return {}
            
            # Apply mask to all furnace data arrays
            result = {}
            for key, values in furnace_data.items():
                if isinstance(values, np.ndarray) and len(values) == len(furnace_mask):
                    result[key] = values[furnace_mask]
                elif key == 'timestamp':  # Handle timestamp specially
                    result[key] = values[furnace_mask] if hasattr(values, '__getitem__') else values
                else:
                    self.logger.debug(f'Skipping non-array furnace data key: {key}')
            
            # Calculate relative time
            if 'timestamp' in result and len(result['timestamp']) > 0:
                result['time'] = (result['timestamp'] - tstart).astype('timedelta64[s]').astype(float)
            
            self.logger.debug(f'Furnace data associated with file {ec_file.fname}, {len(result.get("timestamp", []))} points')
            return result
            
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self.logger.error(f'Error processing furnace data for {ec_file.fname}: {e}')
            return {}
