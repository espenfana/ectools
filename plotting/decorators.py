"""
Plotting decorators for ectools.

This module contains decorators used by plotting methods.
"""

import warnings
from matplotlib import pyplot as plt


def optional_return_figure(func):
    """
    Decorator that adds return_figure parameter to plotting methods.
    
    Plotting methods should always return (fig, ax) tuple where:
    - fig is the matplotlib figure (or None if user provided axes)
    - ax is the axes object
    
    If return_figure=False (default): calls plt.show() only if we created the figure, returns None.
    If return_figure=True: returns the (fig, ax) tuple without showing.
    """
    def wrapper(self, *args, return_figure=False, **kwargs):
        fig, ax = func(self, *args, **kwargs)
        
        if return_figure:
            # Return the figure/axes for further customization
            return fig, ax
        else:
            # Only show if we created the figure (fig is not None)
            if fig is not None:
                plt.show()
                return None
            else:
                # User provided axes, return the axes for their management
                return ax
    
    # Preserve the original function's metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    if hasattr(func, '__annotations__'):
        wrapper.__annotations__ = func.__annotations__
    return wrapper
