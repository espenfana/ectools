#!/usr/bin/env python3
"""
Template and examples for creating derivative electrochemical technique classes.

This module provides templates and examples for creating new technique classes
that combine multiple electrochemical files using the EcList.collate_data() method.
"""

import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

# Import base class (would need proper imports in real usage)
# from ectools.classes.electrochemistry import ElectroChemistry

class ElectroChemistry:
    """Mock base class for template purposes"""
    def __init__(self):
        self.tag = "ElectroChemistry"
        self.meta = {}
        self.aux = {}

class DerivativeTechniqueTemplate(ElectroChemistry):
    """
    Template for creating derivative electrochemical technique classes.
    
    This template shows the basic structure for creating classes that
    combine multiple electrochemical files using EcList.collate_data().
    """
    
    def __init__(self, data_dict: Dict[str, np.ndarray], 
                 aux_dict: Dict[str, Any], 
                 meta_dict: Dict[str, Dict[str, Any]],
                 technique_name: str = "DerivativeTechnique"):
        """
        Initialize derivative technique from collated data.
        
        Args:
            data_dict: Collated data from EcList.collate_data()
            aux_dict: Auxiliary data from EcList.collate_data()
            meta_dict: Metadata from EcList.collate_data()
            technique_name: Name of the technique
        """
        super().__init__()
        
        # Set basic properties
        self.tag = technique_name
        self.technique_name = technique_name
        
        # Copy all data columns as attributes
        for key, value in data_dict.items():
            setattr(self, key, value)
            
        # Store auxiliary and metadata
        self.aux = aux_dict
        self.meta = meta_dict
        
        # Store source information
        self.source_files = list(meta_dict.keys())
        self.n_files = len(self.source_files)
        
        # Set data columns for compatibility
        self.data_columns = list(data_dict.keys())
        
        # Calculate derived properties
        self._calculate_derived_properties()
        
    def _calculate_derived_properties(self):
        """Calculate technique-specific derived properties."""
        # Template method - override in subclasses
        pass
        
    @classmethod
    def from_eclist(cls, eclist, technique_name: str = None, cyclic: bool = False):
        """
        Create derivative technique directly from EcList.
        
        Args:
            eclist: EcList instance (should be pre-filtered to desired files)
            technique_name: Name for the technique
            cyclic: Whether to treat as cyclic data
            
        Returns:
            Instance of the derivative technique class
        """
        if technique_name is None:
            technique_name = cls.__name__
            
        data_dict, aux_dict, meta_dict = eclist.collate_data(
            target_class_name=technique_name,
            cyclic=cyclic
        )
        
        return cls(data_dict, aux_dict, meta_dict, technique_name)
        
    def get_step_data(self, step: int) -> Dict[str, np.ndarray]:
        """Get data for a specific step (original file)."""
        step_mask = self.step == step
        step_data = {}
        
        for col in self.data_columns:
            if hasattr(self, col):
                step_data[col] = getattr(self, col)[step_mask]
                
        return step_data
        
    def get_cycle_data(self, cycle: int) -> Dict[str, np.ndarray]:
        """Get data for a specific cycle."""
        cycle_mask = self.cycle == cycle
        cycle_data = {}
        
        for col in self.data_columns:
            if hasattr(self, col):
                cycle_data[col] = getattr(self, col)[cycle_mask]
                
        return cycle_data
        
    def plot_by_step(self, y_col: str = 'curr', x_col: str = 'time', **kwargs):
        """Plot data colored by step."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for step in sorted(set(self.step)):
            step_data = self.get_step_data(step)
            ax.plot(step_data[x_col], step_data[y_col], 
                   label=f'Step {step}', **kwargs)
                   
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.set_title(f'{self.technique_name} - By Step')
        return fig, ax
        
    def plot_by_cycle(self, y_col: str = 'curr', x_col: str = 'time', **kwargs):
        """Plot data colored by cycle."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cycle in sorted(set(self.cycle)):
            cycle_data = self.get_cycle_data(cycle)
            ax.plot(cycle_data[x_col], cycle_data[y_col], 
                   label=f'Cycle {cycle}', **kwargs)
                   
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.set_title(f'{self.technique_name} - By Cycle')
        return fig, ax

class Electrolysis(DerivativeTechniqueTemplate):
    """
    Electrolysis technique combining CA/CP files.
    
    This class combines multiple chronoamperometry or chronopotentiometry
    files to represent extended electrolysis experiments.
    """
    
    def __init__(self, data_dict: Dict[str, np.ndarray], 
                 aux_dict: Dict[str, Any], 
                 meta_dict: Dict[str, Dict[str, Any]]):
        super().__init__(data_dict, aux_dict, meta_dict, "Electrolysis")
        
    def _calculate_derived_properties(self):
        """Calculate electrolysis-specific properties."""
        if hasattr(self, 'curr') and hasattr(self, 'time'):
            # Total charge passed
            self.total_charge = np.trapz(self.curr, self.time)
            
            # Average current
            self.avg_current = np.mean(self.curr)
            
            # Current efficiency (if multiple steps)
            if hasattr(self, 'step') and len(set(self.step)) > 1:
                self.step_charges = []
                for step in sorted(set(self.step)):
                    step_data = self.get_step_data(step)
                    step_charge = np.trapz(step_data['curr'], step_data['time'])
                    self.step_charges.append(step_charge)
                    
    def faradaic_efficiency(self, theoretical_charge: float) -> float:
        """Calculate Faradaic efficiency."""
        return (self.total_charge / theoretical_charge) * 100
        
    def plot_current_vs_time(self, **kwargs):
        """Plot current vs time for electrolysis."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.time, self.curr, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')
        ax.set_title('Electrolysis Current vs Time')
        
        # Add step boundaries
        if hasattr(self, 'step') and len(set(self.step)) > 1:
            step_changes = np.where(np.diff(self.step) != 0)[0]
            for change_idx in step_changes:
                ax.axvline(self.time[change_idx], color='red', linestyle='--', alpha=0.7)
                
        return fig, ax

class PulsedElectrolysis(DerivativeTechniqueTemplate):
    """
    Pulsed electrolysis technique combining cyclic CA+OCP files.
    
    This class combines alternating chronoamperometry and open circuit
    files to represent pulsed electrolysis experiments.
    """
    
    def __init__(self, data_dict: Dict[str, np.ndarray], 
                 aux_dict: Dict[str, Any], 
                 meta_dict: Dict[str, Dict[str, Any]]):
        super().__init__(data_dict, aux_dict, meta_dict, "PulsedElectrolysis")
        
    def _calculate_derived_properties(self):
        """Calculate pulsed electrolysis properties."""
        if hasattr(self, 'source_tag') and hasattr(self, 'curr'):
            # Separate CA and OCP phases
            self.ca_mask = self.source_tag == 'CA'
            self.ocp_mask = self.source_tag == 'OCP'
            
            # Calculate charge for each cycle's CA phase
            self.cycle_charges = []
            for cycle in sorted(set(self.cycle)):
                cycle_ca_mask = (self.cycle == cycle) & self.ca_mask
                if np.any(cycle_ca_mask):
                    cycle_curr = self.curr[cycle_ca_mask]
                    cycle_time = self.time[cycle_ca_mask]
                    cycle_charge = np.trapz(cycle_curr, cycle_time)
                    self.cycle_charges.append(cycle_charge)
                    
    def plot_pulsed_current(self, **kwargs):
        """Plot pulsed current showing CA and OCP phases."""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot CA phases
        ca_time = self.time[self.ca_mask]
        ca_curr = self.curr[self.ca_mask]
        ax.plot(ca_time, ca_curr, 'b-', label='CA Phase', **kwargs)
        
        # Plot OCP phases (should be near zero current)
        ocp_time = self.time[self.ocp_mask] 
        ocp_curr = self.curr[self.ocp_mask]
        ax.plot(ocp_time, ocp_curr, 'r-', label='OCP Phase', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')
        ax.set_title('Pulsed Electrolysis Current vs Time')
        ax.legend()
        
        return fig, ax
        
    def plot_cycle_comparison(self, y_col: str = 'curr', **kwargs):
        """Plot all cycles overlaid for comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cycle in sorted(set(self.cycle)):
            cycle_data = self.get_cycle_data(cycle)
            # Use relative time within each cycle
            cycle_rel_time = cycle_data['time_rel']
            ax.plot(cycle_rel_time, cycle_data[y_col], 
                   label=f'Cycle {cycle}', **kwargs)
                   
        ax.set_xlabel('Relative Time (s)')
        ax.set_ylabel(y_col)
        ax.legend()
        ax.set_title('Pulsed Electrolysis - Cycle Comparison')
        return fig, ax

def create_custom_technique(technique_name: str, 
                          calculation_func: Optional[callable] = None,
                          plot_func: Optional[callable] = None):
    """
    Factory function to create custom derivative technique classes.
    
    Args:
        technique_name: Name of the new technique
        calculation_func: Function to calculate derived properties
        plot_func: Function for custom plotting
        
    Returns:
        New technique class
    """
    
    class CustomTechnique(DerivativeTechniqueTemplate):
        def __init__(self, data_dict, aux_dict, meta_dict):
            super().__init__(data_dict, aux_dict, meta_dict, technique_name)
            
        def _calculate_derived_properties(self):
            if calculation_func:
                calculation_func(self)
                
        def custom_plot(self, **kwargs):
            if plot_func:
                return plot_func(self, **kwargs)
            else:
                return self.plot_by_step(**kwargs)
                
    CustomTechnique.__name__ = technique_name
    return CustomTechnique

# Example usage
def example_usage():
    """Show example usage of derivative technique classes."""
    
    print("Derivative Technique Classes - Usage Examples")
    print("=" * 50)
    
    print("""
# Example 1: Basic Electrolysis
from ectools import EcList

# Load and filter CA files for electrolysis
eclist = EcList()
eclist.load_files("/path/to/data")
ca_files = eclist.filter(tag='CA')

# Create Electrolysis object
electrolysis = Electrolysis.from_eclist(ca_files)

# Access derived properties
print(f"Total charge: {electrolysis.total_charge:.3f} C")
print(f"Average current: {electrolysis.avg_current:.3f} A")

# Plot results
electrolysis.plot_current_vs_time()
electrolysis.plot_by_step(y_col='volt')
""")

    print("""
# Example 2: Pulsed Electrolysis with cyclic data
# Files: experiment_#1.DTA, experiment_#2.DTA, etc.

# Filter cyclic files
cyclic_files = eclist.filter(fname_contains="experiment_#")

# Create PulsedElectrolysis object with cycle ordering
pulsed = PulsedElectrolysis.from_eclist(cyclic_files, cyclic=True)

# Analyze by cycle
for i, charge in enumerate(pulsed.cycle_charges):
    print(f"Cycle {i} charge: {charge:.3f} C")
    
# Plot pulsed behavior
pulsed.plot_pulsed_current()
pulsed.plot_cycle_comparison()
""")

    print("""
# Example 3: Custom technique class
def calc_impedance(self):
    # Custom calculation for impedance-like technique
    if hasattr(self, 'volt') and hasattr(self, 'curr'):
        self.impedance = self.volt / self.curr
        
def plot_nyquist(self, **kwargs):
    # Custom Nyquist plot
    fig, ax = plt.subplots()
    ax.plot(self.impedance.real, -self.impedance.imag, 'o-', **kwargs)
    ax.set_xlabel('Z_real (Ω)')
    ax.set_ylabel('-Z_imag (Ω)')
    return fig, ax

# Create custom technique
ImpedanceSpectroscopy = create_custom_technique(
    "ImpedanceSpectroscopy",
    calculation_func=calc_impedance,
    plot_func=plot_nyquist
)

# Use it
eis_files = eclist.filter(tag='EIS')
impedance = ImpedanceSpectroscopy.from_eclist(eis_files)
impedance.custom_plot()
""")

if __name__ == "__main__":
    example_usage()
