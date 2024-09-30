'''ectools module'''
from enum import Enum

# ectools root
from .ectools import *
#import ectools.classes as classes

class PlotBackend(Enum):
    '''Selection for plotting backend in ectools'''
    MATPLOTLIB = 'matplotlib'
    BOKEH = 'bokeh'

_config = {
    'plot_backend': PlotBackend.MATPLOTLIB  # default
}

def set_config(key, value):
    '''Set a configuration parameter'''
    if key == 'plot_backend' and not isinstance(value, PlotBackend):
        raise ValueError(f"{value} is not a valid PlotBackend")
    _config[key] = value

def get_config(key):
    """Get a configuration parameter."""
    return _config.get(key)