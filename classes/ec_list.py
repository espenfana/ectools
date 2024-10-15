'''ecList class'''

import pandas as pd
from typing import TypeVar, Generic, List, Dict, Optional

from .electrochemistry import ElectroChemistry

T = TypeVar('T', bound=ElectroChemistry)

class EcList(List[T], Generic[T]):
    '''List class for handling ElectroChemistry class objects'''

    def __init__(self, fpath: Optional[str] = None, **kwargs):
        self.fpath = fpath
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
        '''return a pretty-printable description of EcList contents'''
        describe_df = pd.DataFrame(columns=['idx',
                                            'Filename',
                                            'technique',
                                            'started on',
                                            'finished'])
        for i, f in enumerate(self):
            describe_df.loc[i] = [
                i,
                f.fname,
                f.__class__.__name__,
                f.starttime.strftime('%Y-%m-%d %H:%M:%S'),
                f.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')
            ]
        return describe_df.to_string(index=False)

    def select(self, fids: list = None, **kwargs) -> 'EcList':
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
                file_fid = getattr(file, 'fid', None)  # Handle missing 'fid' gracefully
                if file_fid and file_fid in fids:
                    selected_idx.add(idx)

        # Handle attribute key-value pair filtering (match any key-value pair)
        if kwargs:
            for idx, file in enumerate(self):
                for key, value in kwargs.items():
                    if getattr(file, key, None) == value:
                        selected_idx.add(idx)
                        break  # If one key-value pair matches, add file and break

        if not selected_idx:
            raise ValueError(f"No files found matching the provided criteria: {kwargs or fids}")

        # Create a new EcList instance with the selected files
        selected_files = sorted([self[i] for i in selected_idx], key=lambda f: f.fname, reverse=False)
        eclist_out = EcList(fpath=self.fpath)
        eclist_out.extend(selected_files)
        return eclist_out

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
