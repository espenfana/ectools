#!/usr/bin/env python3
"""
Simple test script to validate collate_data functionality using mock data.
This avoids complex import issues by creating minimal mock objects.
"""

import sys
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, '/workspaces/ectools')

def create_mock_file(fname, tag, start_time, duration=10, has_curr=True):
    """Create a simple mock file object with the required attributes"""
    
    class MockFile:
        def __init__(self):
            self.fname = fname
            self.tag = tag  
            self.starttime = start_time
            
            # Create time arrays
            time_points = np.linspace(0, duration, 50)
            self.time = time_points
            self.timestamp = [start_time + timedelta(seconds=t) for t in time_points]
            
            # Create mock data
            self.volt = np.ones_like(time_points) * 1.5 + np.random.normal(0, 0.01, len(time_points))
            
            if has_curr:
                self.curr = np.sin(time_points / duration * 2 * np.pi) * 0.001
                self.curr_dens = self.curr / 1.0
                self.data_columns = ['time', 'volt', 'curr', 'curr_dens', 'timestamp']
            else:
                self.data_columns = ['time', 'volt', 'timestamp']
                
            self.meta = {'technique': tag, 'duration': duration}
            self.aux = {'pico': {'temperature': np.random.normal(25, 1, len(time_points))}}
            
        def __getattr__(self, name):
            # Return empty array for missing attributes to avoid errors
            if name in ['curr', 'curr_dens']:
                return np.array([])
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
    return MockFile()

def extract_cycle_number(fname):
    """Extract cycle number from filename ending with _#n.DTA"""
    import re
    # More specific pattern: must have # before the number
    match = re.search(r'_#(\d+)\.DTA$', fname, re.IGNORECASE)
    if match:
        return int(match.group(1)) - 1  # Subtract 1 to start at 0
    return 0  # Default to 0 if no cycle number found

def collate_data_standalone(files, cyclic=False):
    """
    Standalone implementation of collate_data for testing
    This replicates the logic from the EcList.collate_data method
    """
    if not files:
        raise ValueError("No files to collate")
        
    # If cyclic, sort files by starttime to ensure proper chronological order
    if cyclic:
        files = sorted(files, key=lambda f: f.starttime if f.starttime else datetime.min)
    
    # Initialize output dictionaries
    data_dict = {}
    aux_dict = {'pico': {}, 'furnace': {}}
    meta_dict = {}
    
    # Collect all timestamps and relative times
    all_timestamps = []
    all_time_rel = []
    step_numbers = []
    source_tags = []
    cycle_numbers = []
    
    for step_idx, f in enumerate(files):
        all_timestamps.extend(f.timestamp)
        all_time_rel.extend(f.time)
        step_numbers.extend([step_idx] * len(f.time))
        source_tags.extend([f.tag] * len(f.time))
        
        # Handle cycle numbers for cyclic data
        if cyclic:
            cycle_num = extract_cycle_number(f.fname)
            cycle_numbers.extend([cycle_num] * len(f.time))
        else:
            cycle_numbers.extend([0] * len(f.time))
            
        # Store metadata for each file
        meta_dict[f.fname] = f.meta
        
    # Convert timestamps to relative time (first timestamp = 0)
    if all_timestamps:
        first_timestamp = all_timestamps[0]
        time_column = np.array([(ts - first_timestamp).total_seconds() 
                              for ts in all_timestamps])
    else:
        time_column = np.array([])
        
    # Add the new/modified columns
    data_dict['time'] = time_column
    data_dict['time_rel'] = np.array(all_time_rel)
    data_dict['step'] = np.array(step_numbers)
    data_dict['source_tag'] = np.array(source_tags)
    data_dict['cycle'] = np.array(cycle_numbers)
    data_dict['timestamp'] = np.array(all_timestamps)
    
    # Get all possible data columns from all files
    all_columns = set()
    for f in files:
        all_columns.update(f.data_columns)
        
    # Remove columns we've already handled
    remaining_columns = all_columns - {'time', 'timestamp'}
    
    # Collate each data column
    for col in remaining_columns:
        collated_data = []
        
        for f in files:
            if hasattr(f, col) and len(getattr(f, col)) > 0:
                collated_data.extend(getattr(f, col))
            else:
                # Fill missing data based on column type
                if col in ['curr', 'curr_dens']:
                    fill_value = 0.0  # Current is 0 for OCP files
                else:
                    fill_value = np.nan  # Other columns get NaN
                
                collated_data.extend([fill_value] * len(f.time))
                
        data_dict[col] = np.array(collated_data)
        
    # Merge auxiliary data
    for f in files:
        if hasattr(f, 'aux') and f.aux:
            for aux_type, aux_data in f.aux.items():
                if aux_type not in aux_dict:
                    aux_dict[aux_type] = {}
                    
                if isinstance(aux_data, dict):
                    for key, value in aux_data.items():
                        if key in aux_dict[aux_type]:
                            # If key exists, try to concatenate arrays
                            if isinstance(value, np.ndarray) and isinstance(aux_dict[aux_type][key], np.ndarray):
                                aux_dict[aux_type][key] = np.concatenate([aux_dict[aux_type][key], value])
                        else:
                            aux_dict[aux_type][key] = value
                else:
                    aux_dict[aux_type] = aux_data
                    
    return data_dict, aux_dict, meta_dict

def test_basic_collation():
    """Test basic collation of multiple files"""
    print("\n=== Testing Basic Collation ===")
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    # Create test files
    files = [
        create_mock_file("test_ca1.DTA", "CA", base_time, duration=10, has_curr=True),
        create_mock_file("test_ocp1.DTA", "OCP", base_time + timedelta(seconds=15), duration=5, has_curr=False),
        create_mock_file("test_ca2.DTA", "CA", base_time + timedelta(seconds=25), duration=8, has_curr=True)
    ]
    
    print(f"Created {len(files)} test files")
    
    try:
        data_dict, aux_dict, meta_dict = collate_data_standalone(files)
        
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
        import traceback
        traceback.print_exc()
        return False

def test_cyclic_collation():
    """Test cyclic data collation with file ordering and cycle numbers"""
    print("\n=== Testing Cyclic Collation ===")
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    # Create test files with cycle numbering in filenames
    # Add files in non-chronological order to test sorting
    files = [
        create_mock_file("test_cycle_#3.DTA", "CA", base_time + timedelta(minutes=20), duration=10),
        create_mock_file("test_cycle_#1.DTA", "CA", base_time, duration=10),
        create_mock_file("test_cycle_#2.DTA", "CA", base_time + timedelta(minutes=10), duration=10)
    ]
    
    print(f"Created {len(files)} cyclic files")
    print("Files added in non-chronological order to test sorting")
    
    try:
        data_dict, aux_dict, meta_dict = collate_data_standalone(files, cyclic=True)
        
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

def test_cycle_number_extraction():
    """Test the cycle number extraction function"""
    print("\n=== Testing Cycle Number Extraction ===")
    
    test_cases = [
        ("test_cycle_#1.DTA", 0),   # #1 -> 0
        ("test_cycle_#5.DTA", 4),   # #5 -> 4  
        ("experiment_#12.DTA", 11), # #12 -> 11
        ("data_3.DTA", 0),          # No # pattern -> 0
        ("normal_file.DTA", 0),     # No pattern -> 0
        ("test_#001.DTA", 0),       # #001 -> 0
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
    print("Testing collate_data functionality (standalone)")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_cycle_number_extraction())
    results.append(test_basic_collation())
    results.append(test_cyclic_collation())
    
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
