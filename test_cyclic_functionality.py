#!/usr/bin/env python3
"""
Test script for the new cyclic parameter in collate_data method
This script properly imports ectools modules to avoid import issues
"""

import sys
import os
from datetime import datetime

# Ensure the ectools directory is in the Python path
sys.path.insert(0, '/workspaces/ectools')

# Import using the package structure
import ectools_main
from helper_functions import mc_filename_parser

def test_cyclic_functionality():
    """Test the cyclic collation functionality"""
    print("=== Testing Cyclic Data Collation Functionality ===\n")
    
    # Set up importer
    imp = ectools_main.EcImporter(fname_parser=mc_filename_parser)
    
    # Test with cyclic data files
    cyclic_fpath = '/workspaces/ectools/testing/cyclic_data'
    regular_fpath = '/workspaces/ectools/testing/data'
    
    # Check if cyclic data directory exists
    if os.path.exists(cyclic_fpath):
        print("1. Testing with cyclic data files:")
        print(f"   Loading from: {cyclic_fpath}")
        
        try:
            fl_cyclic = imp.load_folder(cyclic_fpath)
            print(f"   ✓ Loaded {len(fl_cyclic)} cyclic files")
            
            # Show file details
            print(f"   Files loaded:")
            for i, f in enumerate(fl_cyclic):
                print(f"     {i}: {f.fname} - {f.__class__.__name__}")
                if hasattr(f, 'starttime'):
                    print(f"        Start time: {f.starttime}")
            
            # Test cyclic collation if we have enough files
            if len(fl_cyclic) >= 2:
                print(f"\n   Testing cyclic collation...")
                try:
                    # Test with first few files
                    indices = list(range(min(4, len(fl_cyclic))))
                    data_dict, aux_dict, meta_dict = fl_cyclic.collate_data(
                        indices, "TestCyclicClass", cyclic=True
                    )
                    
                    print(f"     ✓ Cyclic collation successful!")
                    print(f"     Data shape: {data_dict['time'].shape}")
                    print(f"     Columns: {list(data_dict.keys())}")
                    print(f"     Step values: {sorted(set(data_dict['step']))}")
                    print(f"     Cycle values: {sorted(set(data_dict['cycle']))}")
                    print(f"     Source tags: {sorted(set(data_dict['source_tag']))}")
                    
                    # Show file processing order
                    print(f"     Files processed in order:")
                    for step, fname in enumerate(meta_dict.keys()):
                        cycle_data = data_dict['cycle'][data_dict['step'] == step]
                        cycle_num = cycle_data[0] if len(cycle_data) > 0 else 'N/A'
                        print(f"       Step {step}: {fname} (cycle: {cycle_num})")
                        
                except Exception as e:
                    print(f"     ✗ Cyclic collation failed: {e}")
            else:
                print(f"   ⚠ Not enough files for collation test (need ≥2, have {len(fl_cyclic)})")
                
        except Exception as e:
            print(f"   ✗ Failed to load cyclic data: {e}")
    else:
        print(f"1. ⚠ Cyclic data directory not found: {cyclic_fpath}")
    
    print(f"\n" + "="*60 + "\n")
    
    # Test with regular data files
    if os.path.exists(regular_fpath):
        print("2. Testing with regular data files:")
        print(f"   Loading from: {regular_fpath}")
        
        try:
            fl_regular = imp.load_folder(regular_fpath)
            print(f"   ✓ Loaded {len(fl_regular)} regular files")
            
            # Show file details
            print(f"   Files loaded:")
            for i, f in enumerate(fl_regular[:5]):  # Show first 5
                print(f"     {i}: {f.fname} - {f.__class__.__name__}")
            if len(fl_regular) > 5:
                print(f"     ... and {len(fl_regular) - 5} more")
            
            # Test single file conversion
            if len(fl_regular) > 0:
                print(f"\n   Testing single file conversion...")
                try:
                    data_dict, aux_dict, meta_dict = fl_regular.convert_file(
                        0, "TestSingleClass", cyclic=False
                    )
                    print(f"     ✓ Single file conversion successful!")
                    print(f"     Data columns: {list(data_dict.keys())}")
                    print(f"     Data shape: {data_dict['time'].shape}")
                    print(f"     Cycle values: {sorted(set(data_dict['cycle']))}")
                    
                except Exception as e:
                    print(f"     ✗ Single file conversion failed: {e}")
            
            # Test multi-file collation
            if len(fl_regular) >= 3:
                print(f"\n   Testing multi-file collation (non-cyclic)...")
                try:
                    indices = [0, 1, 2]
                    data_dict, aux_dict, meta_dict = fl_regular.collate_data(
                        indices, "TestMultiClass", cyclic=False
                    )
                    print(f"     ✓ Multi-file collation successful!")
                    print(f"     Data shape: {data_dict['time'].shape}")
                    print(f"     Step values: {sorted(set(data_dict['step']))}")
                    print(f"     Cycle values: {sorted(set(data_dict['cycle']))}")
                    print(f"     Source tags: {sorted(set(data_dict['source_tag']))}")
                    
                except Exception as e:
                    print(f"     ✗ Multi-file collation failed: {e}")
            else:
                print(f"   ⚠ Not enough files for multi-file test (need ≥3, have {len(fl_regular)})")
                
        except Exception as e:
            print(f"   ✗ Failed to load regular data: {e}")
    else:
        print(f"2. ⚠ Regular data directory not found: {regular_fpath}")
    
    print(f"\n" + "="*60)
    print("Testing complete!")

if __name__ == "__main__":
    test_cyclic_functionality()
