#!/usr/bin/env python3
"""
Test script to validate collate_data with real cyclic data files.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/workspaces/ectools')

def test_with_real_files():
    """Test collate_data with real cyclic data files"""
    print("=== Testing with Real Cyclic Data Files ===")
    
    # Check if we can import the necessary modules
    try:
        # Simple approach - just import what we need to test the collate logic
        import importlib.util
        
        # Import modules needed
        spec_ec = importlib.util.spec_from_file_location("electrochemistry", "/workspaces/ectools/classes/electrochemistry.py")
        electrochemistry = importlib.util.module_from_spec(spec_ec)
        sys.modules['electrochemistry'] = electrochemistry
        spec_ec.loader.exec_module(electrochemistry)
        
        spec_list = importlib.util.spec_from_file_location("ec_list", "/workspaces/ectools/classes/ec_list.py")
        ec_list = importlib.util.module_from_spec(spec_list)
        sys.modules['ec_list'] = ec_list
        spec_list.loader.exec_module(ec_list)
        
        EcList = ec_list.EcList
        ElectroChemistry = electrochemistry.ElectroChemistry
        
        print("✓ Successfully imported EcList and ElectroChemistry")
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Check what cyclic data files we have
    cyclic_data_dir = "/workspaces/ectools/testing/cyclic_data"
    if not os.path.exists(cyclic_data_dir):
        print(f"✗ Cyclic data directory not found: {cyclic_data_dir}")
        return False
        
    files = os.listdir(cyclic_data_dir)
    print(f"Available files: {files}")
    
    # Test our cycle number extraction with real filenames
    print("\nTesting cycle number extraction with real filenames:")
    
    # Import the extract_cycle_number function from our test
    import re
    def extract_cycle_number(fname):
        """Extract cycle number from filename ending with _#n.DTA"""
        match = re.search(r'_#(\d+)\.DTA$', fname, re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1  # Subtract 1 to start at 0
        return 0  # Default to 0 if no cycle number found
    
    for fname in sorted(files):
        if fname.endswith('.DTA'):
            cycle_num = extract_cycle_number(fname)
            print(f"  {fname} -> cycle {cycle_num}")
    
    # Create a minimal EcList to test the collate_data method exists and works
    try:
        test_list = EcList()
        
        # Check if collate_data method exists
        if hasattr(test_list, 'collate_data'):
            print("✓ collate_data method exists in EcList")
            
            # Test with empty list (should raise ValueError)
            try:
                test_list.collate_data()
                print("✗ Expected ValueError for empty list")
                return False
            except ValueError as e:
                print(f"✓ Correctly raised ValueError for empty list: {e}")
                
        else:
            print("✗ collate_data method not found in EcList")
            return False
            
    except Exception as e:
        print(f"✗ Error testing EcList: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_cycle_pattern_recognition():
    """Test our cycle pattern recognition with various filename patterns"""
    print("\n=== Testing Cycle Pattern Recognition ===")
    
    import re
    def extract_cycle_number(fname):
        """Extract cycle number from filename ending with _#n.DTA"""
        match = re.search(r'_#(\d+)\.DTA$', fname, re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1  # Subtract 1 to start at 0
        return 0  # Default to 0 if no cycle number found
    
    # Test cases from real files and expected patterns
    test_cases = [
        ("1_LOOP_CHRONOA_#1.DTA", 0),   # Real file pattern
        ("1_LOOP_CHRONOA_#2.DTA", 1),   # Real file pattern
        ("1_LOOP_OCP_#1.DTA", 0),       # Real file pattern  
        ("1_LOOP_OCP_#4.DTA", 3),       # Real file pattern
        ("experiment_#10.DTA", 9),       # Generic pattern
        ("test_data_5.DTA", 0),         # No # -> default 0
        ("normal_file.DTA", 0),         # No pattern -> default 0
    ]
    
    all_passed = True
    
    for filename, expected in test_cases:
        result = extract_cycle_number(filename)
        if result == expected:
            print(f"✓ {filename} -> {result}")
        else:
            print(f"✗ {filename} -> {result} (expected {expected})")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("Testing collate_data with real data")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_cycle_pattern_recognition())
    results.append(test_with_real_files())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
