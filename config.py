''' ectools/config.py'''

from enum import Enum

try:
    from bokeh.plotting import figure
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

class Plotter(Enum):
    '''Selection for deafault plotting backend in ectools'''
    MATPLOTLIB = 'matplotlib'
    BOKEH = 'bokeh'

_config = {
    'plotter': Plotter.MATPLOTLIB  # default
}

def set_config(key, value):
    '''Set a configuration parameter'''
    if key == 'plotter' and not isinstance(value, Plotter):
        raise ValueError(f"{value} is not a valid PlotBackend")
    # Attempt to import bokeh
    if value == Plotter.BOKEH:
        if not BOKEH_AVAILABLE:
            raise RuntimeError("Bokeh is not available. Install Bokeh to use this feature.")
    _config[key] = value

def get_config(key):
    """Get a configuration parameter."""
    return _config.get(key)


### Bokeh settings ###
#from bokeh.plotting import ColumnDataSource
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

# Create an instance of BokehSettings to import
if BOKEH_AVAILABLE:
    bokeh_conf = BokehSettings()
