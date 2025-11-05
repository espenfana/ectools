#!/usr/bin/env python3
"""
Usage example for the collate_data functionality in EcList.

This script demonstrates how to use the collate_data method to merge
electrochemical data files into derivative classes like Electrolysis
or PulsedElectrolysis.

Usage example:
1. Load files into EcList
2. Filter/select the files you want to merge
3. Call collate_data() to get merged data dictionaries
4. Use the data to create derivative technique objects
"""

def demonstrate_usage():
    """Demonstrate the collate_data usage workflow"""
    
    print("EcList.collate_data() Usage Example")
    print("=" * 50)
    
    print("""
# Step 1: Load electrochemical data files
from ectools import EcList

# Load your data files
eclist = EcList()
eclist.load_files("/path/to/data")  # Load all .DTA files from directory

# OR load individual files
eclist.append_file("file1.DTA")
eclist.append_file("file2.DTA")

print(f"Loaded {len(eclist)} files")
print(eclist.describe())  # View file details
""")

    print("""
# Step 2: Filter/select files for collation
# Example 1: Select specific files by ID
ca_files = eclist.filter(fids=['001', '003', '005'])

# Example 2: Select by technique type  
ca_files = eclist.filter(tag='CA')

# Example 3: Select files in a range
range_files = eclist.filter(fids=['010', '020'], between=True)

# Example 4: Select files then use the filtered list
selected_files = eclist.filter(tag='CA')  # Get CA files
ocp_files = eclist.filter(tag='OCP')      # Get OCP files
""")

    print("""
# Step 3: Collate data for conversion to derivative classes

# Basic collation (for combining CA/CP into Electrolysis)
data_dict, aux_dict, meta_dict = ca_files.collate_data(
    target_class_name="Electrolysis"
)

# Cyclic collation (for combining cyclic CA+OCP into PulsedElectrolysis)
# This automatically orders files by timestamp and extracts cycle numbers
cyclic_data, cyclic_aux, cyclic_meta = eclist.collate_data(
    target_class_name="PulsedElectrolysis", 
    cyclic=True
)
""")

    print("""
# Step 4: Understand the returned data structure

# data_dict contains all the merged data columns:
print("Data columns:", list(data_dict.keys()))
# Expected: ['time', 'time_rel', 'step', 'source_tag', 'cycle', 'timestamp', 
#           'volt', 'curr', 'curr_dens', ...]

# Key columns:
# - time: unified timeline starting from 0 (seconds from first timestamp)
# - time_rel: original relative time from each file
# - step: file number (0, 1, 2, ...)
# - source_tag: technique tag from each file ('CA', 'OCP', etc.)
# - cycle: cycle number (extracted from filename or 0)
# - timestamp: original datetime stamps

# aux_dict contains merged auxiliary data (temperature, etc.)
print("Auxiliary data:", aux_dict.keys())

# meta_dict contains metadata from each source file
print("Metadata files:", list(meta_dict.keys()))
""")

    print("""
# Step 5: Create derivative technique objects (future implementation)

# Option A: Use existing ElectroChemistry class as base
from ectools.classes.electrochemistry import ElectroChemistry

class Electrolysis(ElectroChemistry):
    def __init__(self, data_dict, aux_dict, meta_dict):
        super().__init__()
        # Set the collated data
        for key, value in data_dict.items():
            setattr(self, key, value)
        self.aux = aux_dict
        self.meta = meta_dict
        
        # Add Electrolysis-specific methods
        self.tag = "Electrolysis" 
        
    def total_charge(self):
        \"\"\"Calculate total charge passed during electrolysis\"\"\"
        from scipy import integrate
        return integrate.trapz(self.curr, self.time)
        
    def average_current_density(self):
        \"\"\"Calculate average current density\"\"\"
        return np.mean(self.curr_dens)

# Create the electrolysis object
electrolysis = Electrolysis(data_dict, aux_dict, meta_dict)
""")

    print("""
# Example: Handling cyclic data with proper ordering

# Files named like: experiment_#1.DTA, experiment_#2.DTA, etc.
# The collate_data method will:
# 1. Sort files chronologically by starttime
# 2. Extract cycle numbers from filenames (_#1 -> cycle 0, _#2 -> cycle 1)
# 3. Create unified timeline across all cycles
# 4. Add cycle column to track which cycle each data point belongs to

cyclic_files = eclist.filter(fname_contains="experiment_#")
pulsed_data, pulsed_aux, pulsed_meta = cyclic_files.collate_data(
    target_class_name="PulsedElectrolysis",
    cyclic=True
)

# Now you can analyze by cycle:
for cycle_num in sorted(set(pulsed_data['cycle'])):
    cycle_mask = pulsed_data['cycle'] == cycle_num
    cycle_current = pulsed_data['curr'][cycle_mask] 
    cycle_time = pulsed_data['time'][cycle_mask]
    print(f"Cycle {cycle_num}: {len(cycle_current)} points")
""")

    print("""
# Step 6: Advanced filtering and analysis

# Combine CA and OCP files in sequence
ca_files = eclist.filter(tag='CA')
ocp_files = eclist.filter(tag='OCP') 

# Merge them in a specific order
combined_list = EcList()
combined_list.extend(ca_files)
combined_list.extend(ocp_files)

# Collate the combined sequence
sequence_data, _, _ = combined_list.collate_data(
    target_class_name="CAOCPSequence"
)

# Analyze the sequence
ca_indices = sequence_data['source_tag'] == 'CA'
ocp_indices = sequence_data['source_tag'] == 'OCP'

print(f"CA data points: {sum(ca_indices)}")
print(f"OCP data points: {sum(ocp_indices)}")
""")

    print("""
# Key Benefits of collate_data():

1. **Unified Timeline**: All files merged into single time axis
2. **Data Alignment**: Missing columns filled appropriately (curr=0 for OCP)
3. **Metadata Preservation**: Original file info maintained
4. **Cycle Support**: Automatic cycle numbering for repeated experiments
5. **Flexibility**: Works with pre-filtered EcList contents
6. **Auxiliary Data**: Temperature and other aux data properly merged

# This enables creation of derivative classes like:
# - Electrolysis (CA/CP files)
# - PulsedElectrolysis (cyclic CA+OCP)
# - GalvanostaticEIS (multiple EIS at different currents) 
# - TemperatureRamp (multiple files at different temperatures)
""")

if __name__ == "__main__":
    demonstrate_usage()
