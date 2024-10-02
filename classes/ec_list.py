'''ecList class'''

import pandas as pd

from .electrochemistry import ElectroChemistry

class EcList(list):
    '''List class for handling ElectroChemistry class objects'''

    def __init__(self, fpath = None, **kwargs):
        self.fpath = fpath
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._fid_idx = {}

    def __repr__(self):
        return f'ecList with {len(self)} items'

    def file_indices(self):
        '''Returns a dict with ecList contents'''
        return {i: item.fname for i, item in enumerate(self)}

    def file_class(self):
        '''Returns a dict with class names for ecList contents'''
        return{i: item.__class__.__name__ for i, item in enumerate(self)}

    def describe(self):
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

    def select(self, fids=None, **kwargs) -> 'EcList':
        """
        Select files based on a list of file IDs and/or attribute key-value pairs.

        Args:
            fids (str or list, optional): A single file ID or a list of file IDs.
            kwargs: Key-value pairs to filter by file attributes.

        Returns:
            EcList: A new instance of EcList containing the selected files.
        """
        # Normalize and ensure fids is a list, even if a single string is passed
        if fids:
            if isinstance(fids, str):
                fids = [fids]
            fids = [self._normalize_fid(fid) for fid in fids]  # Normalize all fids

        # Start by selecting files based on normalized fids
        if fids:
            # Get the indices of files that match the fids
            indices = [self._fid_idx[fid] for fid in fids if fid in self._fid_idx]
        else:
            # If no fids are provided, start with all files
            indices = list(range(len(self)))

        # Filter further by the provided key-value pairs
        selected_files = []
        for idx in indices:
            file = self[idx]
            match = True
            for key, value in kwargs.items():
                if getattr(file, key, None) != value:
                    match = False
                    break
            if match:
                selected_files.append(file)

        if not selected_files:
            raise ValueError(f"No files found matching the provided criteria: {kwargs} and fids: {fids}")

        # Return a new EcList instance with the selected files
        new_list = EcList(fpath=self.fpath)
        new_list.extend(selected_files)
        return new_list

    def select_file(self, fid: str) -> ElectroChemistry:
        """Select a single file based on its unique file ID (fid)."""
        if fid in self._fid_idx:
            return self[self._fid_idx[fid]]
        else:
            raise ValueError(f"File ID '{fid}' not found in the list.")

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

    def fid(self, fid:str) -> ElectroChemistry:
        '''Return file based on file id (number + letter) (must be parsed from e.g. filename)'''
        if fid in self._fid_idx:
            return self[self._fid_idx[fid]]
        raise ValueError(f"File ID '{fid}' not found in the list.")
