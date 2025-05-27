#!/usr/bin/env python3
"""
Test script for the new cyclic parameter in collate_data method
"""

import sys
import os
sys.path.insert(0, '/workspaces/ectools')

# Import directly from the local modules
from ectools_main import EcImporter
from helper_functions import mc_filename_parser

# Test the cyclic functionality
def test_cyclic_collation():
    print("Testing cyclic data collation...")
    
    # Set up importer and load test data
    imp = EcImporter(fname_parser=mc_filename_parser)
    fpath = '/workspaces/ectools/testing/data'
    
    # Load the folder
    fl = imp.load_folder(fpath)
    
    print(f"Loaded {len(fl)} files")
    print("\nFile list:")
    for i, f in enumerate(fl):
        print(f"{i}: {f.fname} - {f.__class__.__name__} - {f.starttime}")
    
    # Test single file conversion
    print("\n=== Testing single file conversion ===")
    if len(fl) > 0:
        data_dict, aux_dict, meta_dict = fl.convert_file(0, "TestClass", cyclic=False)
        print(f"Single file conversion successful")
        print(f"Data columns: {list(data_dict.keys())}")
        print(f"Cycle column shape: {data_dict['cycle'].shape}")
        print(f"Unique cycle values: {set(data_dict['cycle'])}")
    
    # Test multi-file collation with cyclic=False
    print("\n=== Testing multi-file collation (non-cyclic) ===")
    if len(fl) >= 3:
        indices = [0, 1, 2]
        data_dict, aux_dict, meta_dict = fl.collate_data(indices, "TestClass", cyclic=False)
        print(f"Non-cyclic collation successful")
        print(f"Data shape: {data_dict['time'].shape}")
        print(f"Step values: {set(data_dict['step'])}")
        print(f"Cycle values: {set(data_dict['cycle'])}")
        print(f"Source tags: {set(data_dict['source_tag'])}")
    
    # Test multi-file collation with cyclic=True
    print("\n=== Testing multi-file collation (cyclic) ===")
    if len(fl) >= 3:
        indices = [0, 1, 2]
        data_dict, aux_dict, meta_dict = fl.collate_data(indices, "TestClass", cyclic=True)
        print(f"Cyclic collation successful")
        print(f"Data shape: {data_dict['time'].shape}")
        print(f"Step values: {set(data_dict['step'])}")
        print(f"Cycle values: {set(data_dict['cycle'])}")
        print(f"Source tags: {set(data_dict['source_tag'])}")
        
        # Check if files were sorted by starttime
        print(f"Files processed in order:")
        for fname in meta_dict.keys():
            print(f"  {fname}")

if __name__ == "__main__":
    test_cyclic_collation()
