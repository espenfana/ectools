'''ElectrochemicalImpedance class for EIS data'''
from typing import Optional, Dict, Any, Union, Tuple
from matplotlib.axes import Axes
import numpy as np

from .electrochemistry import ElectroChemistry
from ..utils import optional_return_figure


class ElectrochemicalImpedance(ElectroChemistry):
    '''Electrochemical Impedance Spectroscopy (EIS) file container'''

    # Class variables and constants
    _identifiers = {'GALVEIS', 'EISPOT', 'EISMON', 'EISPST', 'EISGAL'}
    # Gamry EIS technique identifiers:
    # - GALVEIS: Galvanostatic EIS
    # - EISPOT: Potentiostatic EIS
    # - EISMON: Single Frequency EIS Monitoring
    
    _column_patterns = {
        **ElectroChemistry._column_patterns,
        'time': (r'^Time$', r'time/(.?s)', r'^T$'),
        'freq': (r'^Freq$', r'Freq\.', r'frequency'),
        'z_real': (r'^Zreal$', r"Z'", r'Re\(Z\)'),
        'z_imag': (r'^Zimag$', r'Z"', r'Im\(Z\)'),
        'z_sig': (r'^Zsig$',),
        'z_mod': (r'^Zmod$', r'\|Z\|', r'Zmod'),
        'z_phase': (r'^Zphz$', r'Phase', r'Zphase'),
        'idc': (r'^Idc$', r'I dc', r'DC Current'),
        'vdc': (r'^Vdc$', r'V dc', r'DC Voltage'),
        'ie_range': (r'^IERange$',),
        'i_mod': (r'^Imod$', r'I mod'),
        'v_mod': (r'^Vmod$', r'V mod'),
        'temp': (r'^Temp$', r'Temperature'),
    }
    
    # Type hints for EIS-specific data columns
    freq: np.ndarray  # Frequency (Hz)
    z_real: np.ndarray  # Real part of impedance (Ohm)
    z_imag: np.ndarray  # Imaginary part of impedance (Ohm)
    z_sig: Optional[np.ndarray]  # Signal flag
    z_mod: np.ndarray  # Impedance modulus (Ohm)
    z_phase: np.ndarray  # Phase angle (degrees)
    idc: Optional[np.ndarray]  # DC current (A)
    vdc: Optional[np.ndarray]  # DC voltage (V)
    ie_range: Optional[np.ndarray]  # Current range
    i_mod: Optional[np.ndarray]  # AC current modulus (A)
    v_mod: Optional[np.ndarray]  # AC voltage modulus (V)
    temp: Optional[np.ndarray]  # Temperature

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        '''Create an ElectrochemicalImpedance file container'''
        super().__init__(*args, **kwargs)
        # Set technique-specific metadata
        self.tag: str = 'EIS'
        self.control: str = 'Impedance'
        
        # Initialize EIS-specific data_columns (will be updated after data load)
        self.data_columns = {
            'time': 'Time (s)',
            'freq': 'Frequency (Hz)',
            'z_real': "Z' (Ω)",
            'z_imag': 'Z" (Ω)',
            'z_mod': '|Z| (Ω)',
            'z_phase': 'Phase (°)',
        }

    def _update_data_columns(self) -> None:
        '''Update data_columns to only include columns that are actually present.
        
        Should be called after data has been loaded to ensure data_columns
        accurately reflects available data.
        '''
        # Define all possible EIS columns with their display names
        possible_columns = {
            'time': 'Time (s)',
            'freq': 'Frequency (Hz)',
            'z_real': "Z' (Ω)",
            'z_imag': 'Z" (Ω)',
            'z_sig': 'Signal',
            'z_mod': '|Z| (Ω)',
            'z_phase': 'Phase (°)',
            'idc': 'DC Current (A)',
            'vdc': 'DC Voltage (V)',
            'ie_range': 'Current Range',
            'i_mod': 'AC Current (A)',
            'v_mod': 'AC Voltage (V)',
            'pot': 'Potential (V)',
            'curr': 'Current (A)',
            'temp': 'Temperature (°C)',
        }
        
        # Only include columns that actually have data
        self.data_columns = {
            col: display_name 
            for col, display_name in possible_columns.items()
            if hasattr(self, col) and getattr(self, col) is not None and len(getattr(self, col)) > 0
        }

    def parse_meta_gamry(self) -> None:
        '''Parse the metadata blocks into attributes for Gamry EIS files'''
        super().parse_meta_gamry()  # Preprocess the metadata block
        
        # EIS-specific metadata parsing
        # Note: self.meta is already a list of lists (split by tabs during file parsing)
        for line in self.meta:
            if not line or not isinstance(line, list):
                continue
            
            # Parse technique-specific parameters
            if line[0] == 'TAG':
                self.technique_tag = line[2] if len(line) > 2 else None
            elif line[0] == 'FREQINIT':
                self.freq_init = float(line[2]) if len(line) > 2 else None
            elif line[0] == 'FREQFINAL':
                self.freq_final = float(line[2]) if len(line) > 2 else None
            elif line[0] == 'PTSPERDEC':
                self.pts_per_decade = float(line[2]) if len(line) > 2 else None
            elif line[0] == 'IACREQ' and len(line) > 2:
                self.i_ac_req = float(line[2]) if line[2] else None
            elif line[0] == 'VACREQ' and len(line) > 2:
                self.v_ac_req = float(line[2]) if line[2] else None
            elif line[0] == 'IDCREQ' and len(line) > 2:
                self.i_dc_req = float(line[2]) if line[2] else None
            elif line[0] == 'VDCREQ' and len(line) > 2:
                self.v_dc_req = float(line[2]) if line[2] else None
            elif line[0] == 'ZGUESS':
                self.z_guess = float(line[2]) if len(line) > 2 else None
        
        # Update data_columns to only include columns that are actually present
        self._update_data_columns()

    def parse_meta_mpt(self) -> None:
        '''Parse the metadata blocks into attributes for EC-Lab EIS files'''
        super().parse_meta_mpt()  # Preprocess the metadata block
        # TODO: Add EC-Lab specific EIS metadata parsing if needed
        
        # Update data_columns to only include columns that are actually present
        self._update_data_columns()

    @optional_return_figure
    def plot(self,
            ax: Optional[Axes] = None,
            plot_type: str = 'auto',
            x: Optional[str] = None,
            y: Optional[str] = None,
            hue: Optional[Union[str, bool]] = None,
            mask: Optional[np.ndarray] = None,
            ax_kws: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Axes:
        '''
        Plot EIS data using matplotlib.
        
        Args:
            ax: Matplotlib axes object. If None, creates new figure.
            plot_type: Type of plot - 'auto' (default), 'nyquist', 'bode_mod', 'bode_phase', or 'custom'
                       'auto' plots z_mod vs time for single-frequency EIS, nyquist otherwise
            x: X-axis data column (for custom plots)
            y: Y-axis data column (for custom plots)
            hue: Column name for color grouping
            mask: Boolean mask for data filtering
            ax_kws: Additional keyword arguments for axes configuration
            **kwargs: Additional arguments passed to pyplot
            
        Returns:
            fig, ax: Figure and axes objects
        '''
        ax_kws = ax_kws or {}
        
        # Auto-detect plot type based on data
        if plot_type == 'auto':
            # Check if this is single-frequency EIS (only one unique frequency)
            if hasattr(self, 'freq') and len(np.unique(self.freq)) == 1:
                # Single frequency monitoring - plot z_mod vs time
                plot_type = 'z_mod_vs_time'
            else:
                # Multi-frequency sweep - default to Nyquist
                plot_type = 'nyquist'
        
        # Helper to handle super().plot() return which may be (fig, ax) or just ax
        def _call_parent_plot(**plot_kwargs):
            result = super(ElectrochemicalImpedance, self).plot(**plot_kwargs, return_figure=True)
            # Handle both return types from parent plot
            if isinstance(result, tuple):
                return result  # (fig, ax)
            else:
                return None, result  # Just ax was returned, no fig created
        
        if plot_type == 'z_mod_vs_time':
            # Single-frequency EIS monitoring: |Z| vs time
            x = 'time'
            y = 'z_mod'
            ax_kws.setdefault('xlabel', 'Time (s)')
            ax_kws.setdefault('ylabel', '|Z| (Ω)')
            ax_kws.setdefault('title', f'Impedance vs Time - {self.fname}')
            fig, ax = _call_parent_plot(ax=ax, x=x, y=y, mask=mask, hue=hue, ax_kws=ax_kws, **kwargs)
            
        elif plot_type == 'nyquist':
            # Nyquist plot: -Zimag vs Zreal
            x = 'z_real'
            y = 'z_imag'
            y_data = -getattr(self, y)  # Negative of imaginary part
            ax_kws.setdefault('xlabel', "Z' (Ω)")
            ax_kws.setdefault('ylabel', "-Z'' (Ω)")
            ax_kws.setdefault('title', f'Nyquist Plot - {self.fname}')
            
            fig, ax_result = _call_parent_plot(ax=ax, x=x, y=y, mask=mask, hue=hue, ax_kws=ax_kws, **kwargs)
            # Override y-data with negative values
            if mask is not None:
                y_data = y_data[mask]
            ax_result.lines[-1].set_ydata(y_data)
            ax_result.set_aspect('equal', adjustable='datalim')
            ax = ax_result
            
        elif plot_type == 'bode_mod':
            # Bode magnitude plot: log|Z| vs log(freq)
            x = 'freq'
            y = 'z_mod'
            ax_kws.setdefault('xlabel', 'Frequency (Hz)')
            ax_kws.setdefault('ylabel', '|Z| (Ω)')
            ax_kws.setdefault('title', f'Bode Plot (Magnitude) - {self.fname}')
            
            fig, ax_result = _call_parent_plot(ax=ax, x=x, y=y, mask=mask, hue=hue, ax_kws=ax_kws, **kwargs)
            ax_result.set_xscale('log')
            ax_result.set_yscale('log')
            ax = ax_result
            
        elif plot_type == 'bode_phase':
            # Bode phase plot: phase vs log(freq)
            x = 'freq'
            y = 'z_phase'
            ax_kws.setdefault('xlabel', 'Frequency (Hz)')
            ax_kws.setdefault('ylabel', 'Phase (°)')
            ax_kws.setdefault('title', f'Bode Plot (Phase) - {self.fname}')
            
            fig, ax_result = _call_parent_plot(ax=ax, x=x, y=y, mask=mask, hue=hue, ax_kws=ax_kws, **kwargs)
            ax_result.set_xscale('log')
            ax = ax_result
            
        else:  # custom plot
            if x is None or y is None:
                raise ValueError("For custom plots, both x and y must be specified")
            fig, ax = _call_parent_plot(ax=ax, x=x, y=y, mask=mask, hue=hue, ax_kws=ax_kws, **kwargs)
        
        if hue:
            ax.legend(title=hue)
        
        # Ensure we always return (fig, ax) tuple for consistency with decorator
        # If ax was provided by user, fig will be None
        return fig, ax

    @optional_return_figure
    def plot_nyquist(self,
                    ax: Optional[Axes] = None,
                    **kwargs: Any) -> Axes:
        '''Convenience method for Nyquist plot'''
        return self.plot(ax=ax, plot_type='nyquist', return_figure=True, **kwargs)

    @optional_return_figure
    def plot_bode(self,
                 ax: Optional[Axes] = None,
                 **kwargs: Any) -> Tuple[Axes, Axes]:
        '''
        Convenience method for combined Bode plot (magnitude and phase)
        
        Returns two axes objects (magnitude, phase)
        '''
        if ax is not None:
            raise ValueError("For combined Bode plot, ax must be None (creates new figure)")
        
        from matplotlib import pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        # Plot magnitude (top panel)
        ax1.plot(self.freq, self.z_mod, **kwargs)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('|Z| (Ω)')
        ax1.set_title(f'Bode Plot (Magnitude) - {self.fname}')
        ax1.legend()
        
        # Plot phase (bottom panel)
        ax2.plot(self.freq, self.z_phase, **kwargs)
        ax2.set_xscale('log')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (°)')
        ax2.set_title(f'Bode Plot (Phase) - {self.fname}')
        ax2.legend()
        
        plt.tight_layout()
        
        return fig, (ax1, ax2)
