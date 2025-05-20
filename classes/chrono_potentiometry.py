'''ChronoPotiometry class'''
from .electrochemistry import ElectroChemistry
# ChronoAmperometry Class
class ChronoPotentiometry(ElectroChemistry):
    '''Chronopotentiometry file container'''

    # Class variables and constants
    identifiers = {'Chronopotentiometry', 'CP', 'CHRONOP'}
    # Data columns to import
    column_patterns = {**ElectroChemistry.column_patterns,
    }
    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.match.
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict

    # Initialize
    def __init__(self, *args, **kwargs):
        '''Create a Chronopotentiometry file container'''
        super().__init__(*args, **kwargs)
        self.time_step = float()
        self.curr_step = float()
        self.tag = 'CP'
        self.control = 'Amperostatic'

    # Class methods
    def parse_meta_mpt(self):
        '''Parse the metadata blocks into attributes'''
        super().parse_meta_mpt() # Preprocess the metadata block
        self.curr_step = 'NotImplemented'

    def parse_meta_gamry(self):
        '''Parse the metadata list into attributes'''
        super().parse_meta_gamry()
        self.time_step = float()
        self.curr_step = float()

    def plot(self,
            ax=None,
            x='time',
            y='pot',
            hue=None,
            mask=None,
            ax_kws = None,
            **kwargs):
        '''Plot data using matplotlib. Any kwargs are passed along to pyplot'''
        ax = super().plot(ax=ax, x=x, y=y, mask=mask, hue=hue, ax_kws = ax_kws, **kwargs)
        if hue:
            ax.legend(title=hue)
