'''ecList class'''
class EcList(list):
    '''List class for handling ElectroChemistry class objects'''

    def __init__(self, fpath = None, **kwargs):
        self.fpath = fpath
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return f'ecList with {len(self)} items'

    def file_indices(self):
        '''Returns a dict with ecList contents'''
        return {i: item.fname for i, item in enumerate(self)}

    def file_class(self):
        '''Returns a dict with class names for ecList contents'''
        return{i: item.__class__.__name__ for i, item in enumerate(self)}
