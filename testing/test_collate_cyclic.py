#!/usr/bin/env python3
"""
Test script for the new cyclic parameter in collate_data method
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, '/workspaces/ectools')

# Import the classes directly
from classes.ec_list import EcList
from classes.electrochemistry import ElectroChemistry
from classes.cyclic_voltammetry import CyclicVoltammetry
from classes.chrono_amperometry import ChronoAmperometry  
from classes.open_circuit import OpenCircuit
from ectools_main import EcImporter
from helper_functions import mc_filename_parser

# Test the cyclic functionality
def test_cyclic_collation():
    print("Testing cyclic data collation...")
    
    # Set up importer and load test data
    imp = EcImporter(fname_parser=mc_filename_parser)
    
    # Test with the new cyclic data
    cyclic_fpath = '/workspaces/ectools/testing/cyclic_data'
    regular_fpath = '/workspaces/ectools/testing/data'
    
    print("=== Testing with cyclic data files ===")
    fl_cyclic = imp.load_folder(cyclic_fpath)
    
    print(f"Loaded {len(fl_cyclic)} cyclic files")
    print("\nCyclic file list:")
    for i, f in enumerate(fl_cyclic):
        print(f"{i}: {f.fname} - {f.__class__.__name__} - {f.starttime}")
    
    # Test cyclic collation with the actual cyclic files
    if len(fl_cyclic) >= 4:
        print("\n=== Testing cyclic collation with real cyclic files ===")
        # Get indices for OCP files (should have cycle numbers)
        ocp_indices = [i for i, f in enumerate(fl_cyclic) if 'OCP' in f.fname]
        
        if len(ocp_indices) >= 2:
            indices = ocp_indices[:4]  # Take first 4 OCP files
            print(f"Using OCP files at indices: {indices}")
            
            data_dict, aux_dict, meta_dict = fl_cyclic.collate_data(indices, "PulsedElectrolysis", cyclic=True)
            print(f"Cyclic collation successful")
            print(f"Data shape: {data_dict['time'].shape}")
            print(f"Step values: {set(data_dict['step'])}")
            print(f"Cycle values: {set(data_dict['cycle'])}")
            print(f"Source tags: {set(data_dict['source_tag'])}")
            
            # Check if files were sorted by starttime and cycle numbers extracted
            print(f"\nFiles processed in chronological order:")
            for i, fname in enumerate(meta_dict.keys()):
                cycle_num = data_dict['cycle'][data_dict['step'] == i][0] if len(data_dict['cycle'][data_dict['step'] == i]) > 0 else 'N/A'
                print(f"  Step {i}: {fname} (extracted cycle: {cycle_num})")
    
    # Also test with regular data
    print("\n=== Testing with regular data files ===")
    fl_regular = imp.load_folder(regular_fpath)
    
    print(f"Loaded {len(fl_regular)} regular files")
    
    # Test single file conversion
    if len(fl_regular) > 0:
        print("\n=== Testing single file conversion ===")
        data_dict, aux_dict, meta_dict = fl_regular.convert_file(0, "TestClass", cyclic=False)
        print(f"Single file conversion successful")
        print(f"Data columns: {list(data_dict.keys())}")
        print(f"Cycle column shape: {data_dict['cycle'].shape}")
        print(f"Unique cycle values: {set(data_dict['cycle'])}")
    
    # Test multi-file collation with cyclic=False
    if len(fl_regular) >= 3:
        print("\n=== Testing multi-file collation (non-cyclic) ===")
        indices = [0, 1, 2]
        data_dict, aux_dict, meta_dict = fl_regular.collate_data(indices, "TestClass", cyclic=False)
        print(f"Non-cyclic collation successful")
        print(f"Data shape: {data_dict['time'].shape}")
        print(f"Step values: {set(data_dict['step'])}")
        print(f"Cycle values: {set(data_dict['cycle'])}")
        print(f"Source tags: {set(data_dict['source_tag'])}")

if __name__ == "__main__":
    test_cyclic_collation()
