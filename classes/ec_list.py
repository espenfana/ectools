'''ecList class'''
import warnings
from collections.abc import Iterable
from typing import TypeVar, Generic, List, Dict, Optional

import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

from .electrochemistry import ElectroChemistry

T = TypeVar('T', bound=ElectroChemistry)

class EcList(List[T], Generic[T]):
    '''List class for handling ElectroChemistry class objects'''

    def __init__(self, fpath: Optional[str] = None, **kwargs):
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

    def filter(self, fids: list = None, sorting = None, **kwargs) -> 'EcList':
        """
        Select files based on a list of file IDs (fids) OR matching any attribute key-value pair.
        Files are selected if they match any of the file IDs or any of the key-value pairs.

        Args:
            fids : A list of file IDs (fids).
            kwargs: Key-value pairs to filter by file attributes.

        Returns:
            EcList: A new instance of EcList containing the selected files.
        """
        selected_idx = set()

        # Handle file ID (fid) filtering using the 'fid' attribute of each file
        if fids:
            fids = [fid.lower().lstrip('0') for fid in fids]  # Normalize fids
            for idx, file in enumerate(self):
                file_fid = getattr(file, 'id', None)  # Handle missing 'fid' gracefully
                if file_fid and file_fid in fids:
                    selected_idx.add(idx)

        # Handle attribute key-value pair filtering (match any key-value pair)
        if kwargs:
            for idx, file in enumerate(self):
                for key, value in kwargs.items():
                    file_attr = getattr(file, key, None)
                    if isinstance(value, Iterable) and not isinstance(value, str):
                        if file_attr in value:
                            selected_idx.add(idx)
                            break  # If one key-value pair matches, add file and break
                    else:
                        if file_attr == value:
                            selected_idx.add(idx)
                            break  # If one key-value pair matches, add file and break

        if not selected_idx:
            raise ValueError(f"No files found matching the provided criteria: {kwargs or fids}")

        # Create a new EcList instance with the selected files
        selected_files = sorted([self[i] for i in selected_idx],
                                key=lambda f: f.fname,
                                reverse=False)
        eclist_out = EcList(fpath=self.fpath)
        eclist_out.extend(selected_files)
        if sorting:
            eclist_out.sort(key=lambda f: getattr(f, sorting), reverse=False)
        return eclist_out

    def select(self, fid: str = None, **kwargs) -> T:
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

    def plot(self, group = None, titles='fname', **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to f.plot()'''
        if group and getattr(self[0], group):
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

    def _generate_fid_idx(self):
        """Generates a dictionary mapping normalized file ID to index."""
        for i, f in enumerate(self):
            id_number = getattr(f, 'id_number', None)
            id_letter = getattr(f, 'id_letter', '') or ''  # Ensure id_letter is a string
            id_letter = id_letter.lower()  # Normalize letter case
            if id_number:
                key = self._normalize_fid(str(id_number) + id_letter)
                self._fid_idx[key] = i

    def _normalize_fid(self, fid):
        """Normalize the file ID by removing leading zeroes and converting to lowercase."""
        return fid.lstrip('0').lower()

    def fid(self, fid:str) -> ElectroChemistry: # deprecated
        '''Return file based on file id (number + letter) (must be parsed from e.g. filename)'''
        if fid in self._fid_idx:
            return self[self._fid_idx[fid]]
        raise ValueError(f"File ID '{fid}' not found in the list.")
