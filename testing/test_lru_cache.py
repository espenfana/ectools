"""
Test LRU cache functionality for ectools.

This test validates that the new LRU cache strategy properly handles
multiple cache files with different parameter combinations.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import ectools
# Navigate up from testing/ -> ectools/ -> work/ectools/ (repository root parent)
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

import ectools as ec
from ectools.helper_functions import example_filename_parser


def test_lru_cache_multiple_params():
    """Test that multiple caches are kept with different parameters."""
    print("\n=== Test 1: Multiple caches with different parameters ===")
    
    # Setup
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data'
    
    # Set cache to keep 3 files
    ec.set_max_cache_files(3)
    
    # Create importer with different aux_data_classes to create different cache hashes
    imp1 = ec.EcImporter(fname_parser=example_filename_parser, log_level="INFO")
    imp2 = ec.EcImporter(fname_parser=None, log_level="INFO")  # Different parser
    
    # Clear any existing cache first
    cache_dir = imp1._get_cache_dir(str(data_path))
    for cache_file in cache_dir.glob('*.pkl'):
        cache_file.unlink()
    for cache_file in cache_dir.glob('*.json'):
        cache_file.unlink()
    
    # Load with first configuration
    print("\nLoading with config 1 (fname_parser=example_filename_parser)...")
    eclist1 = imp1.load_folder(str(data_path))
    time.sleep(0.1)  # Small delay to ensure different mtimes
    
    # Load with second configuration
    print("\nLoading with config 2 (fname_parser=None)...")
    eclist2 = imp2.load_folder(str(data_path))
    time.sleep(0.1)
    
    # Check that we have 2 cache files
    cache_files = list(cache_dir.glob('*.pkl'))
    print(f"\nCache files after 2 loads: {len(cache_files)}")
    assert len(cache_files) == 2, f"Expected 2 cache files, got {len(cache_files)}"
    
    print("✓ Test 1 passed: Multiple caches are kept")


def test_lru_cache_eviction():
    """Test that oldest cache is deleted when exceeding max_cache_files."""
    print("\n=== Test 2: LRU eviction when exceeding limit ===")
    
    # Setup
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data'
    
    # Set cache to keep only 2 files
    ec.set_max_cache_files(2)
    
    # Clear existing cache
    imp = ec.EcImporter(fname_parser=example_filename_parser, log_level="INFO")
    cache_dir = imp._get_cache_dir(str(data_path))
    for cache_file in cache_dir.glob('*.pkl'):
        cache_file.unlink()
    for cache_file in cache_dir.glob('*.json'):
        cache_file.unlink()
    
    # Create 3 different configurations
    imp1 = ec.EcImporter(fname_parser=example_filename_parser, log_level="INFO")
    imp2 = ec.EcImporter(fname_parser=None, log_level="INFO")
    
    # Load with first config
    print("\nLoading with config 1...")
    eclist1 = imp1.load_folder(str(data_path))
    cache_files_before = list(cache_dir.glob('*.pkl'))
    cache1_name = cache_files_before[0].name
    print(f"Cache 1 created: {cache1_name}")
    time.sleep(0.1)
    
    # Load with second config
    print("\nLoading with config 2...")
    eclist2 = imp2.load_folder(str(data_path))
    cache_files = list(cache_dir.glob('*.pkl'))
    cache2_name = [f.name for f in cache_files if f.name != cache1_name][0]
    print(f"Cache 2 created: {cache2_name}")
    print(f"Cache files: {len(cache_files)}")
    assert len(cache_files) == 2, f"Expected 2 cache files, got {len(cache_files)}"
    time.sleep(0.1)
    
    # Load with first config again (should create third cache and delete oldest)
    print("\nLoading with config 1 again (should evict cache 1)...")
    eclist1_again = imp1.load_folder(str(data_path))
    cache_files_after = list(cache_dir.glob('*.pkl'))
    print(f"Cache files after 3rd load: {len(cache_files_after)}")
    
    # Should still have 2 cache files (cache1 deleted, new cache created)
    assert len(cache_files_after) == 2, f"Expected 2 cache files after eviction, got {len(cache_files_after)}"
    
    print("✓ Test 2 passed: Oldest cache was evicted")


def test_lru_cache_ordering():
    """Test that LRU ordering is based on modification time."""
    print("\n=== Test 3: LRU ordering based on modification time ===")
    
    # Setup
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data'
    
    # Set cache to keep 3 files
    ec.set_max_cache_files(3)
    
    # Clear existing cache
    imp = ec.EcImporter(fname_parser=example_filename_parser, log_level="INFO")
    cache_dir = imp._get_cache_dir(str(data_path))
    for cache_file in cache_dir.glob('*.pkl'):
        cache_file.unlink()
    for cache_file in cache_dir.glob('*.json'):
        cache_file.unlink()
    
    # Create 3 different configurations
    imp1 = ec.EcImporter(fname_parser=example_filename_parser, log_level="INFO")
    imp2 = ec.EcImporter(fname_parser=None, log_level="INFO")
    
    # Load with configs and track order
    print("\nCreating 3 caches with delays...")
    eclist1 = imp1.load_folder(str(data_path))
    time.sleep(0.2)
    
    eclist2 = imp2.load_folder(str(data_path))
    time.sleep(0.2)
    
    # Re-load config 1 (should update its mtime, making it newest)
    eclist1_reload = imp1.load_folder(str(data_path))
    
    # Get cache files with their mtimes
    cache_files = [(f, f.stat().st_mtime) for f in cache_dir.glob('*.pkl')]
    cache_files.sort(key=lambda x: x[1])  # Sort by mtime (oldest first)
    
    print(f"\nCache files in order (oldest to newest):")
    for cf, mtime in cache_files:
        print(f"  {cf.name}: {mtime}")
    
    # The oldest should be from imp2, newest should be from imp1
    assert len(cache_files) == 2, f"Expected 2 cache files, got {len(cache_files)}"
    
    print("✓ Test 3 passed: LRU ordering is correct")


def test_max_cache_files_config():
    """Test that set_max_cache_files configuration works."""
    print("\n=== Test 4: Configuration changes ===")
    
    # Test default value
    default_max = ec.get_max_cache_files()
    print(f"Default max_cache_files: {default_max}")
    assert default_max == 3, f"Expected default 3, got {default_max}"
    
    # Change to 5
    ec.set_max_cache_files(5)
    new_max = ec.get_max_cache_files()
    print(f"After set_max_cache_files(5): {new_max}")
    assert new_max == 5, f"Expected 5, got {new_max}"
    
    # Change to 1 (old behavior)
    ec.set_max_cache_files(1)
    old_behavior = ec.get_max_cache_files()
    print(f"After set_max_cache_files(1): {old_behavior}")
    assert old_behavior == 1, f"Expected 1, got {old_behavior}"
    
    # Reset to default
    ec.set_max_cache_files(3)
    
    print("✓ Test 4 passed: Configuration changes work correctly")


def test_data_path_txt_not_deleted():
    """Test that data_path.txt is never deleted."""
    print("\n=== Test 5: data_path.txt is preserved ===")
    
    # Setup
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data'
    
    # Set cache to keep only 1 file
    ec.set_max_cache_files(1)
    
    # Create importer and get cache dir
    imp = ec.EcImporter(fname_parser=example_filename_parser, log_level="INFO")
    cache_dir = imp._get_cache_dir(str(data_path))
    
    # data_path.txt should exist after cache dir creation
    ref_file = cache_dir / 'data_path.txt'
    assert ref_file.exists(), "data_path.txt should exist"
    
    # Load folder (creates cache)
    eclist = imp.load_folder(str(data_path))
    
    # data_path.txt should still exist
    assert ref_file.exists(), "data_path.txt should not be deleted"
    
    # Load with different config (should delete old cache but keep data_path.txt)
    imp2 = ec.EcImporter(fname_parser=None, log_level="INFO")
    eclist2 = imp2.load_folder(str(data_path))
    
    # data_path.txt should still exist
    assert ref_file.exists(), "data_path.txt should still exist after cache cleanup"
    
    print("✓ Test 5 passed: data_path.txt is preserved")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing LRU Cache Functionality")
    print("="*70)
    
    try:
        test_max_cache_files_config()
        test_lru_cache_multiple_params()
        test_lru_cache_eviction()
        test_lru_cache_ordering()
        test_data_path_txt_not_deleted()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
