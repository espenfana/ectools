'''ectools module'''
from enum import Enum

# ectools root
from .ectools_main import *
from . import classes
from .bokeh_settings import bokeh
from . import helper_functions

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

# Attempt to import bokeh
try:
    from bokeh.plotting import figure
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
