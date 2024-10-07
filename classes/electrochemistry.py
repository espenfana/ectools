'''Parent electrochemistry file class'''
import re
from datetime import datetime, timedelta

import dateutil.parser as date_parser
from matplotlib import pyplot as plt
import numpy as np

import ectools as ec
from ectools import BOKEH_AVAILABLE

if BOKEH_AVAILABLE:
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource

class ElectroChemistry():
    ''' The default container and parent class for containing electrochemistry files and methods
    '''
    # Class variables and constants
    identifiers = set()

    # Data columns to be imported. Keys will become instance attributes so must adhere to a strict
    # naming scheme. The values should be list-like to support multiple different regex identifiers,
    # which are used in a re.search.
    get_columns = {
        'redherring': (r'redherring',), # An identifier which is not found will not generate errors
        'signal': (r'Sig',), # Signal, i.e. target potenital
        'time': (r'time/(.?s)',r'^T$',), # Time column
        'pot': (r'<?Ewe>?/(.?V)', r'potential', r'^Vf$',), # Potential column
        'curr':(r'<?I>?/(.?A)', r'^Im$')} # Current column
    # Use (group) to search for the unit. the last (groups) in the regex will be added to a dict

    # Initialize
    def __init__(self, fname, fpath, meta, **kwargs):
        ''' Create a generalized ElecroChemistry object'''
        self.fname = fname # Filename
        self.fpath = fpath # Path to file
        self.meta = meta # Metadata block
        for key, val in kwargs.items():
            setattr(self, key, val)
        # Initialize data columns as empty arrays
        self.time = np.empty(0)
        self.curr = np.empty(0)
        self.curr_dens = np.empty(0)
        self.pot = np.empty(0)
        self.timestamps = np.empty(0)
        self.units = {}
        self._meta_dict = {}
        # These should remain empty in this class
        #self.cycle = np.empty(0)
        #self.oxred = np.empty(0)

        self.area = float()
        self.starttime = datetime
        self.label = None
    def __getitem__(self, key):
        '''Makes object subscriptable like a dict'''
        return self.__getattribute__(key)
    def __setitem__(self, key, value):
        '''Makes object attributes assignable like a dict'''
        self.__setattr__(key, value)
    def __repr__(self):
        return self.__class__.__name__ + 'object from file' + self.fname

    # Class methods
    def parse_meta_mpt(self):
        '''Parse attributes from the metadata block'''
        #self.colon_delimited = [row.split(':') for row in self.meta]
        colonsep = {} # For colon sepatated metadata
        wsep = [] # For width separated metadata, which may include
        # 20 pt splitter and stripper replaced by function split_by_length
        #split20 = lambda s: [s.strip()] if len(s) < 20 else [s[:20].strip(), *split20(s[20:])]
        for row in self.meta:
            if (':' in row) and (row[18:20] != '  '):
                colonsep[row.split(':', 1)[0].strip()] = row.split(':', 1)[1].strip()
            elif (len(row)>20) and (row[18:20] == '  '):
                if re.match(r'vs\.', row):
                    wsep[-1].append(_split_by_length(row[20:]))
                else:
                    m = re.search(r'\((.?\w)\)', row[:20]) # look for units in parenthesis
                    if m:
                        wsep.append([row[:m.start(0)].strip(),
                                     _split_by_length(row[20:]), m.group(1)])
                    else:
                        wsep.append([row[:20].strip(), 
                                     _split_by_length(row[20:])])
        # If the experiment was modified during run, the last value will be entered
        widthsep = {row[0]: row[1:] for row in wsep}
        self._meta_dict = {**colonsep, **widthsep}
        self.starttime = date_parser.parse(self._meta_dict['Acquisition started on'])

    def parse_meta_gamry(self):
        '''parse attribudes from the metadata block'''
        self._meta_dict = {}
        i = 0
        while i < len(self.meta):
            line = self.meta[i]
            match len(line):
                case 2:
                    self._meta_dict[line[0]] = {'value': line[1]}
                case 4 | 5:
                    if line[0] == 'NOTES': # handle the multi-line note field
                        self._meta_dict[line[0]] = {'value': ""}
                        i += 1
                        for _ in range(i,int(line[2])):
                            self._meta_dict[line[0]]['value'] += self.meta[i]
                            i += 1
                    else:
                        self._meta_dict[line[0]] = {
                            'label': line[1],
                            'value': line[2],
                            'description': ', '.join(line[3:])
                        }
            i += 1
        # Import into attributes
        metamap = {'area': 'AREA'}
        for key, label in metamap.items():
            self[key] = float(self._meta_dict[label]['value'])
            try:
                self.units[key] = re.search(r'\((.*?)\)',
                                            self._meta_dict[label]['description']).group(1)
            except Exception:
                pass

        date_str = self._meta_dict['DATE']['value']
        time_str = self._meta_dict['TIME']['value']
        self.starttime = date_parser.parse(date_str + ' ' + time_str)
        timedeltas = np.array([timedelta(seconds=t) for t in self.time])
        self.timestamps = np.array([self.starttime + delta for delta in timedeltas])

    def makelab(self, axid):
        '''Generate an axis label with unit'''
        d = {'curr': 'I ', 'pot': 'E ', 'time': 'time ', 'curr_dens': 'I\''}
        return d[axid] + '(' + self.units[axid] + ')'

    def plot_bokeh(self, x='time', y='curr'): #TODO further work
        """Plot using Bokeh with global settings."""
        if not ec.BOKEH_AVAILABLE:
            raise RuntimeError("Bokeh is not available. Install Bokeh to use this feature.")

        # Generate default tooltips if none are set
        #ec.bokeh.generate_default_tooltips(self.data.keys())

        # Use Bokeh settings for figure size, title, and tooltips
        p = figure(
            width=ec.bokeh.figsize[0],  # Use the configured figure size
            height=ec.bokeh.figsize[1],
            title=ec.bokeh.title,
        )
        p.add_tools(ec.bokeh.hover)
        p.line(x,y,source=self._create_column_data_source())
        show(p)

    def _create_column_data_source(self) ->ColumnDataSource :
        source = ColumnDataSource(
            data={
                'time' : self.time,
                'pot' : self.pot,
                'curr' : self.curr,
                'curr_dens' : self.curr_dens,
                'timestamp' : self.timestamps
            }
        )
        return source

    def plot(self,
        ax=None, # pyplot axes
        x='time', # key for x axis array
        y='curr', # key for y axis array
        color = 'tab:blue', # color
        hue = None, # split the plot based on values in a third array
        clause = None, # logical array to slice the arrays
        ax_kws = None, # arguments passed to ax.set()
        **kwargs):
        '''Plot data using matplotlib.
            Parameters are seaborn-like. Any additional kwargs are passed along to pyplot'''
        ax_kws = ax_kws or {}
        if not ax:
            _, ax = plt.subplots()
        if not clause:
            clause = np.full(self[x].shape, True)
        if hue is True: # no hue set by technique
            hue = False
        if hue:
            for val in np.unique(self[hue][clause]):
                ax.plot(
                    self[x][self[hue]==val],
                    self[y][self[hue]==val],
                    label = str(val),
                     **kwargs)
        else:
            ax.plot(
                self[x][clause],
                self[y][clause],
                color = color,
                **kwargs)
        if 'xlabel' not in ax_kws:
            ax_kws['xlabel'] = self.makelab(x)
        if 'ylabel' not in ax_kws:
            ax_kws['ylabel'] = self.makelab(y)
        ax.set(**ax_kws) # Set axes properties, such as xlabel etc.
        return ax

    def plotyy(self,
            fig = None,
            x = 'time', # key for the common x-axis
            y_left = 'pot', # key for left y-axis
            color_left = 'tab:blue', # color for left y-axis
            y_right = 'curr', # key for right y-axis
            color_right = 'tab:red', # color for right y-axis
            hue = None, # split the plot based on values in a third array
            clause = None, # logical array to slice the arrays
            ax_left_kws = None, # arguments passed to ax.set()
            ax_right_kws = None, # arguments passed to ax.set()
            **kwargs):
        '''Plot data with two y-scales using matplotlib. 
            Parameters are seaborn-like. Any additional kwargs are passed along to pyplot'''
        if not fig:
            fig = plt.figure()
        ax_left = fig.add_subplot(111)
        ax_left = self.plot(ax=ax_left,
                            x=x,
                            y=y_left,
                            color=color_left,
                            hue=hue,
                            clause=clause,
                            ax_kws=ax_left_kws,
                            **kwargs)

        ax_right = ax_left.twinx()
        ax_right = self.plot(ax=ax_right,
                             x=x,
                             y=y_right,
                             color=color_right,
                             hue=hue,
                             clause=clause,
                             ax_kws=ax_right_kws,
                             **kwargs)

        ax_left.spines['left'].set_color(color_left)
        ax_right.spines['right'].set_color(color_right)
        return fig, (ax_left, ax_right)

    def set_area(self, new_area: float):
        '''Set new value for area and recalculate current density
            new_area: float [cm²]'''
        self.area = new_area
        self.curr_dens = self.curr / new_area
        self.units['curr_dens'] = f'{self.units["curr"]}/cm²'

def _split_by_length(s, length=20):
    """
    Splits the input string `s` into chunks of `length`, stripping whitespace from each chunk.
    
    Parameters:
    - s: The input string to be split.
    - length: The length of each chunk (default is 20).
    
    Returns:
    - A list of stripped chunks of the input string.
    """
    result = []
    while s:
        result.append(s[:length].strip())
        s = s[length:]
    return result
