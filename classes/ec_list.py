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

    def generate_fid_idx(self):
        '''Runs once to generate fid to index dict'''
        for i, f in enumerate(self):
            if f.id_number:
                if f.id_letter:
                    self._fid_idx[str(f.id_number) + str(f.id_letter)] = i
                else:
                    self._fid_idx[str(f.id_number)] = i

    def fid(self, fid:str) -> ElectroChemistry:
        '''Return file based on file id (number + letter) (must be parsed from e.g. filename)'''
        if fid in self._fid_idx:
            return self[self._fid_idx[fid]]
        else:
            raise ValueError(f"File ID '{fid}' not found in the list.")
