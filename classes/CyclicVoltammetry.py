from .ElectroChemistry import ElectroChemistry, ec
# CyclicVoltammetry Class
class CyclicVoltammetry(ElectroChemistry):
    '''Cyclic voltammetry file container'''

    # Class variables and constants
    identifiers = {'Cyclic Voltammetry'} # Strings in the raw files which indicate the technique
    get_columns = {**ElectroChemistry.get_columns,
        'oxred': (r'ox/red',), 
        'cycle': (r'cycle number',),
        }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict naming scheme. The values should be list-like to support multiple different regex identifiers, which are used in a re.match.    
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict
    
    
    # Initialize

    def __init__(self, *args, **kwargs):
        '''Create a Cyclic Voltammetry container'''
        super().__init__(*args, **kwargs)
        

    # Class methods
    def parse_meta_mpt(self):
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        self.scanrate = float(self.widthsep['dE/dt'][0][0])
        self.units['scanrate'] = self.widthsep['dE/dt unit'][0]
        self.pot_init = float(self.widthsep['Ei'][0][0])
        self.units['pot_init'] = self.widthsep['Ei'][1]
        self.pot_upper = float(self.widthsep['E1'][0][0])
        self.units['pot_upper'] = self.widthsep['E1'][1]
        self.pot_lower = float(self.widthsep['E2'][0][0])
        self.units['pot_lower'] = self.widthsep['E2'][1]
        self.pot_end = float(self.widthsep['Ef'][0][0])
        self.units['pot_end'] = self.widthsep['Ef'][1]
        self.ncycles = int(self.widthsep['nc cycles'][0][0])

    def plot(self, 
    ax=None, 
    x='pot', 
    y='curr', 
    color='tab:blue', 
    cycles=None, 
    clause=None, 
    hue=None, 
    ax_kws={}, 
    **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        if cycles:
            clause = ec.np.where(ec.np.logical_and(self['cycle']>=cycles[0],  self['cycle']<=cycles[1]))
        ax = super().plot(ax=ax, x=x, y=y, clause=clause, hue=hue, ax_kws=ax_kws, **kwargs)
        if hue:
            ax.legend(title=hue)
        return ax