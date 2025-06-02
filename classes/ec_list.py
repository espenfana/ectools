'''ecList class'''
import warnings
from collections.abc import Iterable
from typing import TypeVar, Generic, List, Dict, Optional, Any

import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

from .electrochemistry import ElectroChemistry

T = TypeVar('T', bound=ElectroChemistry)

class EcList(List[T], Generic[T]):
    '''List class for handling ElectroChemistry class objects'''
    
    def __init__(self, fpath: Optional[str] = None, **kwargs: Any) -> None:
        self.fpath = fpath
        self.aux = {'pico': {}, 'furnace': {}}
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._fid_idx: Dict[str, int] = {}

    def __repr__(self) -> str:
        return f'ecList with {len(self)} items'

    def file_indices(self) -> Dict[int, str]:
        '''Returns a dict with ecList contents'''
        return {i: item.fname for i, item in enumerate(self)}

    def file_class(self) -> Dict[int, str]:
        '''Returns a dict with class names for ecList contents'''
        return {i: item.__class__.__name__ for i, item in enumerate(self)}

    def describe(self) -> str:
        """
        Return a pretty-printable description of EcList contents.

        This method generates a DataFrame containing details about each element
        in the EcList, including the index, filename, technique, start time, 
        and finish time. The DataFrame is then converted to a string for 
        pretty-printable output.

        Returns:
            describe (str): A pretty-printable string representation of the item.
        """
        tformat = '%Y-%m-%d %H:%M:%S'
        describe_df = pd.DataFrame(columns=['idx',
                                            'Filename',
                                            'technique',
                                            'started on',
                                            'finished'])
        for i, f in enumerate(self):
            finished = f.timestamp[-1].strftime(tformat) if len(f.timestamp) > 0 else None
            describe_df.loc[i] = [
                i,
                f.fname,
                f.__class__.__name__,
                f.starttime.strftime(tformat),
                finished
            ]
        return describe_df.to_string(index=False)
    
    def area_corrections(self, corrections_map: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """
        Apply area corrections to the files in the EcList.

        This method modifies the `area` attribute of the `ElectroChemistry` objects in the list.

        Args:
            corrections_map (Optional[Dict[str, Dict[str, float]]]): A dictionary containing area correction
                values. If not provided, the method will use the 'electrode_corrections' from the aux attribute.

        Raises:
            ValueError: If no area corrections are found or if an invalid electrode format is encountered.
        """

        if corrections_map:
            # If corrections_map is provided, it will be used directly.
            pass
        elif 'electrode_corrections' in self.aux:
            corrections_map = self.aux['electrode_corrections']
        else:
            raise ValueError('No area corrections found in aux, and none provided')
        try:
            for electrode, corr in corrections_map.items():
                if electrode.startswith("WE") and electrode[2:].isdigit():
                    electrode_number = int(electrode[2:])
                else:
                    raise ValueError(f"Invalid electrode format: {electrode}")
                if corr['length_mm'] is None or corr['width_mm'] is None:
                    continue
                length = corr['length_mm']/10 # Convert to cm
                d = corr['width_mm']/10 # Convert to cm
                area = np.pi * d * length + np.pi * (d/2)**2 # Area of end of cylinder
                print(f"Setting electrode {electrode_number} area: {area:.3f} cm^2")
                for f in self:
                    if f.we_number == electrode_number:
                        f.set_area(area)
        except Exception as e:
            raise ValueError(f"Error applying area corrections: {str(e)}") from e    
        
    def filter(self, fids: Optional[List[str]] = None, sorting: Optional[str] = None, between: bool = False, **kwargs: Any) -> 'EcList':
        """
        Select files based on a list of file IDs (fids), a range if between_fids=True and fids has 2 items,
        OR matching any attribute key-value pair. Files are selected if they match any of the file IDs or
        any of the key-value pairs.

        Args:
            fids : A list of file IDs (fids).
            sorting: An attribute name to sort by.
            between_fids: Whether to select files in the inclusive range of two fids.
            kwargs: Key-value pairs to filter by file attributes.

        Returns:
            EcList: A new instance of EcList containing the selected files.
        """
        selected_idx = set()

        # Check if between_fids=True but fids doesn't have length 2
        if between and (not fids or len(fids) != 2):
            raise ValueError("When between_fids=True, exactly two fids must be provided.")

        if fids:
            normalized_fids = [fid.lower().lstrip('0') for fid in fids]
            if between:
                start, end = sorted([int(nf) for nf in normalized_fids])
                for idx, file in enumerate(self):
                    file_fid = getattr(file, 'id', None)
                    if file_fid and file_fid.isdigit():
                        numeric_id = int(file_fid)
                        if start <= numeric_id <= end:
                            selected_idx.add(idx)
            else:
                for idx, file in enumerate(self):
                    file_fid = getattr(file, 'id', None)
                    if file_fid and file_fid.lower().lstrip('0') in normalized_fids:
                        selected_idx.add(idx)

        if kwargs:
            for idx, file in enumerate(self):
                for key, value in kwargs.items():
                    file_attr = getattr(file, key, None)
                    if isinstance(value, Iterable) and not isinstance(value, str):
                        if file_attr in value:
                            selected_idx.add(idx)
                            break
                    else:
                        if file_attr == value:
                            selected_idx.add(idx)
                            break

        if not selected_idx:
            raise ValueError(f"No files found matching the provided criteria: {kwargs or fids}")

        selected_files = sorted([self[i] for i in selected_idx],
                                key=lambda f: f.fname,
                                reverse=False)
        eclist_out = EcList(fpath=self.fpath)
        eclist_out.extend(selected_files)
        if sorting:
            eclist_out.sort(key=lambda f: getattr(f, sorting), reverse=False)
        return eclist_out    
    
    def select(self, fid: Optional[str] = None, **kwargs: Any) -> T:
        """
        Select a single file based on a file ID (fid) OR matching all attribute key-value pair.
        Files are selected if they match the file ID or all of the key-value pairs.

        Args:
            fid : A file ID (fid).
            kwargs: Key-value pairs to filter by file attributes.

        Returns:
            T: The selected file.

        Raises:
            ValueError: If no files match the criteria.
        """
        selected_files = []
        if fid: 
            selected_files = self.filter(fids=[fid])
        elif kwargs:
            selected_files = self
            for key, value in kwargs.items():
                selected_files = selected_files.filter(**{key: value})

        if len(selected_files) > 1:        
            warnings.warn(
                f"Multiple files found matching the provided criteria: {kwargs or fid}. "
                "Returning the first match."
            )
        # Return the first selected file
        return selected_files[0]

    def plot(self, 
             group: Optional[str] = None, 
             merge: bool = False, 
             titles: Optional[str] = 'fname', 
             **kwargs: Any) -> None:
        '''Plot data using matplotlib. Any kwargs are passed along to f.plot()'''
        if merge:
            _, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
            for f in self:
                f.plot(ax=ax, **kwargs)
                if titles:
                    ax.set_title(getattr(f, titles,''))
        elif group and getattr(self[0], group):
            unique_groups = {getattr(f, group) for f in self}
            for g in unique_groups:
                fl_group = self.filter(**{group:g})
                ncols = len(fl_group)
                _, ax = plt.subplots(1, ncols, figsize=(5*ncols, 5), constrained_layout=True)
                if ncols == 1:
                    ax = [ax]
                for i, f in enumerate(fl_group):
                    f.plot(ax=ax[i], **kwargs)
                    if titles:
                        ax[i].set_title(getattr(f, titles,''))
        else:
            for f in self:
                _, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
                f.plot(ax=ax,**kwargs)
                if titles:
                    ax.set_title(getattr(f, titles,''))

    def _generate_fid_idx(self) -> None:
        """Generates a dictionary mapping normalized file ID to index."""
        for i, f in enumerate(self):
            id_number = getattr(f, 'id_number', None)
            id_letter = getattr(f, 'id_letter', '') or ''  # Ensure id_letter is a string
            id_letter = id_letter.lower()  # Normalize letter case
            if id_number:
                key = self._normalize_fid(str(id_number) + id_letter)
                self._fid_idx[key] = i

    def _normalize_fid(self, fid: str) -> str:
        """Normalize the file ID by removing leading zeroes and converting to lowercase."""
        return fid.lstrip('0').lower()

    def fid(self, fid:str) -> ElectroChemistry: # deprecated
        '''Return file based on file id (number + letter) (must be parsed from e.g. filename)'''
        if fid in self._fid_idx:
            return self[self._fid_idx[fid]]
        raise ValueError(f"File ID '{fid}' not found in the list.")
    
    def collate_data(self, target_class_name=None, cyclic=False):
        """
        Collate data from files in the current EcList for conversion to derivative classes.
        
        Args:
            target_class_name: str - name of target class (for reference/debugging)
            cyclic: bool - if True, files are treated as cyclic data and ordered by starttime,
                          with cycle numbers extracted from filenames and cycle order validated
            
        Returns:
            tuple: (data_dict, aux_dict, meta_dict)
                - data_dict: dict with collated data columns including:
                  * time: unified timeline from timestamps (first timestamp = 0)
                  * time_rel: original relative time from each file
                  * step: file number in the sequence (0-based)
                  * substep: relative step number within each cycle (0-based)
                  * cycle: cycle number (extracted from filename for cyclic data)
                  * source_tag: technique tag from source file
                  * timestamp: original timestamp data
                  * [original data columns]: curr, pot, etc. with missing values filled
                - aux_dict: dict with merged auxiliary data  
                - meta_dict: dict with metadata from each source file
        """
        if not self:
            raise ValueError("No files in EcList to collate")
            
        # Get all files from the current EcList
        files = list(self)
        
        # If cyclic, sort files by starttime to ensure proper chronological order
        if cyclic:
            files.sort(key=lambda f: f.starttime if f.starttime else datetime.min)
        
        # Initialize output dictionaries
        data_dict = {}
        aux_dict = {'pico': {}, 'furnace': {}}
        meta_dict = {}
        
        # Determine total length and collect timestamps for time calculation
        total_length = sum(len(f.time) for f in files)
        all_timestamps = []
        all_time_rel = []
        step_numbers = []
        source_tags = []
        cycle_numbers = []  # New for cyclic data
        
        # Collect all timestamps, relative times, and cycle information
        import re
        file_cycle_numbers = []  # Track cycle numbers by file for validation
        
        for step_idx, f in enumerate(files):
            all_timestamps.extend(f.timestamp)
            all_time_rel.extend(f.time)
            step_numbers.extend([step_idx] * len(f.time))
            source_tags.extend([f.tag] * len(f.time))
            
            # Handle cycle numbers for cyclic data
            if cyclic:
                # Inline cycle extraction: pattern must have # before the number
                match = re.search(r'_#(\d+)\.DTA$', f.fname, re.IGNORECASE)
                if match:
                    cycle_num = int(match.group(1)) - 1  # Subtract 1 to start at 0
                else:
                    cycle_num = 0  # Default to 0 if no cycle number found
                    print(f"Warning: No cycle number found in filename {f.fname}, defaulting to cycle 0")
                
                file_cycle_numbers.append(cycle_num)
                cycle_numbers.extend([cycle_num] * len(f.time))
            else:
                cycle_numbers.extend([0] * len(f.time))  # Default to 0 for non-cyclic
            
            # Store metadata for each file
            meta_dict[f.fname] = f.meta
            
        # Validate cycle order for cyclic data
        if cyclic and len(set(file_cycle_numbers)) > 1:
            sorted_cycles = sorted(file_cycle_numbers)
            expected_cycles = list(range(min(file_cycle_numbers), max(file_cycle_numbers) + 1))
            if file_cycle_numbers != sorted_cycles:
                print(f"Warning: Files are not in cycle order. Found cycles: {file_cycle_numbers}, Expected order: {sorted_cycles}")
            if sorted_cycles != expected_cycles:
                missing_cycles = set(expected_cycles) - set(file_cycle_numbers)
                if missing_cycles:
                    print(f"Warning: Missing cycles detected: {missing_cycles}")
                duplicate_cycles = [x for x in file_cycle_numbers if file_cycle_numbers.count(x) > 1]
                if duplicate_cycles:
                    print(f"Warning: Duplicate cycles detected: {set(duplicate_cycles)}")
        
        # Create substep column (relative step number within each cycle)
        substep_numbers = []
        if cyclic:
            # Group steps by cycle and assign substep numbers starting from 0
            cycle_step_counts = {}
            for step_idx, cycle_num in zip(step_numbers, cycle_numbers):
                if cycle_num not in cycle_step_counts:
                    cycle_step_counts[cycle_num] = {}
                if step_idx not in cycle_step_counts[cycle_num]:
                    cycle_step_counts[cycle_num][step_idx] = len(cycle_step_counts[cycle_num])
                substep_numbers.append(cycle_step_counts[cycle_num][step_idx])
        else:
            # For non-cyclic data, substep is the same as step
            substep_numbers = step_numbers.copy()
            
        # Convert timestamps to relative time (first timestamp = 0)
        if all_timestamps:
            first_timestamp = all_timestamps[0]
            time_column = np.array([(ts - first_timestamp).total_seconds() 
                                  for ts in all_timestamps])
        else:
            time_column = np.array([])
            
        # Add the new/modified columns
        data_dict['time'] = time_column
        data_dict['time_rel'] = np.array(all_time_rel)
        data_dict['step'] = np.array(step_numbers)
        data_dict['substep'] = np.array(substep_numbers)  # Add substep column
        data_dict['source_tag'] = np.array(source_tags)
        data_dict['cycle'] = np.array(cycle_numbers)  # Add cycle column
        data_dict['timestamp'] = np.array(all_timestamps)
        
        # Get all possible data columns from all files
        all_columns = set()
        for f in files:
            all_columns.update(f.data_columns)
            
        # Remove columns we've already handled
        remaining_columns = all_columns - {'time', 'timestamp'}
        
        # Collate each data column
        for col in remaining_columns:
            collated_data = []
            
            for f in files:
                if hasattr(f, col) and len(getattr(f, col)) > 0:
                    collated_data.extend(getattr(f, col))
                else:
                    # Fill missing data based on column type
                    if col in ['curr', 'curr_dens']:
                        # Current is 0 for OCP files
                        fill_value = 0.0
                    else:
                        # Other columns get NaN
                        fill_value = np.nan
                    
                    collated_data.extend([fill_value] * len(f.time))
                    
            data_dict[col] = np.array(collated_data)
            
        # Merge auxiliary data
        for f in files:
            if hasattr(f, 'aux') and f.aux:
                for aux_type, aux_data in f.aux.items():
                    if aux_type not in aux_dict:
                        aux_dict[aux_type] = {}
                        
                    if isinstance(aux_data, dict):
                        for key, value in aux_data.items():
                            if key in aux_dict[aux_type]:
                                # If key exists, try to concatenate arrays
                                if isinstance(value, np.ndarray) and isinstance(aux_dict[aux_type][key], np.ndarray):
                                    aux_dict[aux_type][key] = np.concatenate([aux_dict[aux_type][key], value])
                                # For non-arrays, keep the first occurrence
                            else:
                                aux_dict[aux_type][key] = value
                    else:
                        # Handle non-dict auxiliary data
                        aux_dict[aux_type] = aux_data
                        
        return data_dict, aux_dict, meta_dict

    def convert_file(self, index, target_class_name=None, cyclic=False):
        """
        Convenience method to convert a single file.
        
        Args:
            index: int - index of file in EcList to convert
            target_class_name: str - name of target class (for reference/debugging)
            cyclic: bool - if True, file is treated as cyclic data
            
        Returns:
            tuple: (data_dict, aux_dict, meta_dict) - same as collate_data
        """
        return self.collate_data([index], target_class_name, cyclic)
