#!/usr/bin/env python3
"""
Test script to validate the collate_data functionality in EcList.

This script tests:
1. Basic collation of multiple files
2. Cyclic data handling with file ordering and cycle numbers
3. Data column merging and missing value handling
4. Time alignment and auxiliary data merging
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, '/workspaces/ectools')
sys.path.insert(0, '/workspaces/ectools/classes')

# Try different import strategies
try:
    # First try importing from the package
    import ectools
    from ectools.classes.ec_list import EcList
    print("✓ Successfully imported from ectools package")
except ImportError:
    try:
        # Try direct imports
        from classes.ec_list import EcList
        from classes.electrochemistry import ElectroChemistry  
        from classes.chrono_amperometry import ChronoAmperometry
        from classes.open_circuit import OpenCircuit
        print("✓ Successfully imported classes directly")
    except ImportError:
        try:
            # Try importing as modules
            import importlib.util
            
            # Load ec_list module
            spec = importlib.util.spec_from_file_location("ec_list", "/workspaces/ectools/classes/ec_list.py")
            ec_list_module = importlib.util.module_from_spec(spec)
            
            # Load electrochemistry module first (dependency)
            electrochem_spec = importlib.util.spec_from_file_location("electrochemistry", "/workspaces/ectools/classes/electrochemistry.py")
            electrochem_module = importlib.util.module_from_spec(electrochem_spec)
            sys.modules['electrochemistry'] = electrochem_module
            electrochem_spec.loader.exec_module(electrochem_module)
            
            # Now load ec_list
            sys.modules['ec_list'] = ec_list_module
            spec.loader.exec_module(ec_list_module)
            
            EcList = ec_list_module.EcList
            ElectroChemistry = electrochem_module.ElectroChemistry
            print("✓ Successfully imported using importlib")
        except Exception as e:
            print(f"✗ All import strategies failed: {e}")
            sys.exit(1)

def create_mock_ca_file(fname, fid, start_time, duration=10, has_curr=True):
    """Create a mock ChronoAmperometry file for testing"""
    print(f"Creating mock CA file: {fname}")
    
    # Create time arrays
    time_points = np.linspace(0, duration, 100)
    timestamps = [start_time + timedelta(seconds=t) for t in time_points]
    
    # Create mock data
    if has_curr:
        curr = np.sin(time_points / duration * 2 * np.pi) * 0.001  # Mock current data
        curr_dens = curr / 1.0  # Assuming 1 cm² area
    else:
        curr = np.zeros_like(time_points)  # No current for OCP-like files
        curr_dens = np.zeros_like(time_points)
    
    volt = np.ones_like(time_points) * 1.5 + np.random.normal(0, 0.01, len(time_points))  # Mock voltage
    
    # Create mock file object
    mock_file = ChronoAmperometry()
    mock_file.fname = fname
    mock_file.id = fid
    mock_file.tag = 'CA'
    mock_file.starttime = start_time
    mock_file.time = time_points
    mock_file.timestamp = timestamps
    mock_file.volt = volt
    mock_file.curr = curr
    mock_file.curr_dens = curr_dens
    mock_file.data_columns = ['time', 'volt', 'curr', 'curr_dens', 'timestamp']
    mock_file.meta = {'technique': 'Chronoamperometry', 'duration': duration}
    mock_file.aux = {'pico': {'temperature': np.random.normal(25, 1, len(time_points))}}
    
    return mock_file

def create_mock_ocp_file(fname, fid, start_time, duration=5):
    """Create a mock OpenCircuit file for testing"""
    print(f"Creating mock OCP file: {fname}")
    
    # Create time arrays
    time_points = np.linspace(0, duration, 50)
    timestamps = [start_time + timedelta(seconds=t) for t in time_points]
    
    # OCP files typically don't have current data
    volt = np.ones_like(time_points) * 1.2 + np.random.normal(0, 0.005, len(time_points))
    
    # Create mock file object
    mock_file = OpenCircuit()
    mock_file.fname = fname
    mock_file.id = fid
    mock_file.tag = 'OCP'
    mock_file.starttime = start_time
    mock_file.time = time_points
    mock_file.timestamp = timestamps
    mock_file.volt = volt
    mock_file.data_columns = ['time', 'volt', 'timestamp']  # No current columns
    mock_file.meta = {'technique': 'Open Circuit', 'duration': duration}
    mock_file.aux = {'pico': {'temperature': np.random.normal(25, 1, len(time_points))}}
    
    return mock_file

def test_basic_collation():
    """Test basic collation of multiple files"""
    print("\n=== Testing Basic Collation ===")
    
    # Create test files
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    eclist = EcList()
    
    # Add some CA and OCP files
    eclist.append(create_mock_ca_file("test_ca1.DTA", "001", base_time, duration=10))
    eclist.append(create_mock_ocp_file("test_ocp1.DTA", "002", base_time + timedelta(seconds=15), duration=5))
    eclist.append(create_mock_ca_file("test_ca2.DTA", "003", base_time + timedelta(seconds=25), duration=8))
    
    print(f"Created EcList with {len(eclist)} files")
    
    # Test collation
    try:
        data_dict, aux_dict, meta_dict = eclist.collate_data(target_class_name="TestCollation")
        
        print("✓ Collation completed successfully")
        print(f"  Data columns: {list(data_dict.keys())}")
        print(f"  Total data points: {len(data_dict['time'])}")
        print(f"  Time range: {data_dict['time'][0]:.2f} to {data_dict['time'][-1]:.2f} seconds")
        print(f"  Step range: {data_dict['step'].min()} to {data_dict['step'].max()}")
        print(f"  Source tags: {set(data_dict['source_tag'])}")
        print(f"  Metadata files: {list(meta_dict.keys())}")
        
        # Check that missing current data was filled with zeros
        ocp_indices = data_dict['source_tag'] == 'OCP'
        if np.any(ocp_indices):
            ocp_currents = data_dict['curr'][ocp_indices]
            if np.allclose(ocp_currents, 0):
                print("✓ OCP current data correctly filled with zeros")
            else:
                print("✗ OCP current data not correctly filled")
        
        return True
        
    except Exception as e:
        print(f"✗ Collation failed: {e}")
        return False

def test_cyclic_collation():
    """Test cyclic data collation with file ordering and cycle numbers"""
    print("\n=== Testing Cyclic Collation ===")
    
    # Create test files with cycle numbering in filenames
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    eclist = EcList()
    
    # Add files in non-chronological order to test sorting
    # Cycle 2 (should be ordered second)
    eclist.append(create_mock_ca_file("test_cycle_#3.DTA", "003", base_time + timedelta(minutes=20), duration=10))
    # Cycle 0 (should be ordered first)  
    eclist.append(create_mock_ca_file("test_cycle_#1.DTA", "001", base_time, duration=10))
    # Cycle 1 (should be ordered second)
    eclist.append(create_mock_ca_file("test_cycle_#2.DTA", "002", base_time + timedelta(minutes=10), duration=10))
    
    print(f"Created EcList with {len(eclist)} cyclic files")
    print("Files added in non-chronological order to test sorting")
    
    # Test cyclic collation
    try:
        data_dict, aux_dict, meta_dict = eclist.collate_data(target_class_name="CyclicTest", cyclic=True)
        
        print("✓ Cyclic collation completed successfully")
        print(f"  Total data points: {len(data_dict['time'])}")
        print(f"  Cycle numbers: {sorted(set(data_dict['cycle']))}")
        print(f"  Steps: {sorted(set(data_dict['step']))}")
        
        # Check cycle number extraction
        expected_cycles = [0, 1, 2]  # Should be 0, 1, 2 (subtract 1 from filename numbers)
        actual_cycles = sorted(set(data_dict['cycle']))
        if actual_cycles == expected_cycles:
            print("✓ Cycle numbers correctly extracted and ordered")
        else:
            print(f"✗ Cycle numbers incorrect. Expected {expected_cycles}, got {actual_cycles}")
            
        # Check chronological ordering by examining time continuity
        steps = data_dict['step']
        times = data_dict['time']
        
        # Check that each step's time continues from the previous
        for step in range(1, max(steps) + 1):
            prev_step_mask = steps == (step - 1)
            curr_step_mask = steps == step
            
            if np.any(prev_step_mask) and np.any(curr_step_mask):
                prev_max_time = times[prev_step_mask].max()
                curr_min_time = times[curr_step_mask].min()
                
                if curr_min_time > prev_max_time:
                    print(f"✓ Step {step} correctly follows step {step-1} in time")
                else:
                    print(f"✗ Step {step} time ordering issue")
        
        return True
        
    except Exception as e:
        print(f"✗ Cyclic collation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_filtered_collation():
    """Test collation after filtering EcList"""
    print("\n=== Testing Filtered Collation ===")
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    # Create original EcList with many files
    eclist = EcList()
    eclist.append(create_mock_ca_file("test_ca1.DTA", "001", base_time, duration=10))
    eclist.append(create_mock_ocp_file("test_ocp1.DTA", "002", base_time + timedelta(seconds=15), duration=5))
    eclist.append(create_mock_ca_file("test_ca2.DTA", "003", base_time + timedelta(seconds=25), duration=8))
    eclist.append(create_mock_ocp_file("test_ocp2.DTA", "004", base_time + timedelta(seconds=35), duration=5))
    
    print(f"Created EcList with {len(eclist)} files")
    
    # Filter to only CA files
    try:
        ca_files = eclist.filter(tag='CA')
        print(f"Filtered to {len(ca_files)} CA files")
        
        # Collate the filtered list
        data_dict, aux_dict, meta_dict = ca_files.collate_data(target_class_name="FilteredCA")
        
        print("✓ Filtered collation completed successfully")
        print(f"  Data points: {len(data_dict['time'])}")
        print(f"  Source tags: {set(data_dict['source_tag'])}")
        print(f"  Steps: {sorted(set(data_dict['step']))}")
        
        # Verify only CA files were included
        if set(data_dict['source_tag']) == {'CA'}:
            print("✓ Only CA files included in filtered collation")
        else:
            print(f"✗ Unexpected source tags: {set(data_dict['source_tag'])}")
            
        # Verify step numbering (should be 0, 1 for two CA files)
        expected_steps = [0, 1]
        actual_steps = sorted(set(data_dict['step']))
        if actual_steps == expected_steps:
            print("✓ Step numbering correct for filtered collation")
        else:
            print(f"✗ Step numbering incorrect. Expected {expected_steps}, got {actual_steps}")
        
        return True
        
    except Exception as e:
        print(f"✗ Filtered collation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing EcList.collate_data functionality")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_basic_collation())
    results.append(test_cyclic_collation())
    results.append(test_filtered_collation())
    
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
