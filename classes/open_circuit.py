'''Open Circuit class'''
from .electrochemistry import ElectroChemistry

class OpenCircuit(ElectroChemistry):
    '''Open Circuit Voltage file container'''

    # Class variables and constants
    identifiers = {'Open Circuit', 'CORPOT'}
    column_patterns = {**ElectroChemistry.column_patterns,
    }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict

    # Initialize
    #def __init__(self, *args, **kwargs):
    #    '''Create a Open Circuit type file container'''
    #    super().__init__(*args, **kwargs)

    # Class methods
    #def parse_meta_mpt(self):
    #    '''Parse the metadata blocks into attributes'''
    #    super().parse_meta_mpt() # Preprocess the metadata block

    def plot(self,
            ax=None,
            x='time',
            y='pot',
            color = 'tab:blue', # color
            hue=None,
            clause=None,
            ax_kws = None,
            **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        kws = {'xlabel': f'time ({self.units[x]})',
                'ylabel': f'E_OC ({self.units[y]})'}
        ax = super().plot(ax=ax, x=x, y=y, clause=clause, hue=hue, ax_kws = kws, **kwargs)
        if hue:
            ax.legend(title=hue)
