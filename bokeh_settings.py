''' ectools/bokeh_settings.py'''
from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool
#from bokeh.io import output_notebook, output_file


NOTEBOOK = 'notebook'
FILE = 'file'

class BokehSettings:
    """Class to handle Bokeh-specific plotting settings."""
    def __init__(self):
        # Default Bokeh settings
        self.figsize = [800, 600]  # Default figure size (width, height)
        self.tooltips = None        # Default tooltips
        self.title = "Bokeh Plot"   # Default plot title
        self.hover = None
        self.output = NOTEBOOK
        
    def set(self, 
            figsize=None,
            tooltips=None,
            title=None,
            output=None):
        """Set Bokeh-specific plot settings."""
        if figsize:
            self.figsize = figsize
        if tooltips:
            self.tooltips = tooltips
        if title:
            self.title = title
        if output:
            if output in [NOTEBOOK, FILE]:
                self.output = output
            else:
                raise ValueError(f'Output method {output} not recognized')

    def get(self, setting):
        """Retrieve a specific Bokeh setting."""
        return getattr(self, setting, None)

    def set_tooltips(self):
        '''create hover object and set properties'''
        self.hover = HoverTool()
        self.hover.tooltips = [
        ('E (V)', '@pot'),
        ('I (A)', '@curr'),
        ('time (s)', '@time'),
        ('timestamp', '@timestamps{%F %T}'),
        ('cycle', '@cycle')
        ]
        self.hover.formatters={'@timestamps': 'datetime'}

# Create a global instance of BokehSettings
bokeh = BokehSettings()
bokeh.set_tooltips()
