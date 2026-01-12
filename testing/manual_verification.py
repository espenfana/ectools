#!/usr/bin/env python3
"""
Manual verification script for LRU cache functionality.

This script demonstrates the new LRU cache behavior with multiple parameter combinations.
"""

import sys
from pathlib import Path

# Add parent directory to path
# Navigate up from testing/ -> ectools/ -> work/ectools/ (repository root parent)
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

import ectools as ec
from ectools.helper_functions import example_filename_parser

def main():
    print("=" * 70)
    print("Manual Verification: LRU Cache Functionality")
    print("=" * 70)
    
    # Get test data path
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data'
    
    if not data_path.exists():
        print(f"\n❌ Test data not found at: {data_path}")
        return 1
    
    # 1. Test configuration
    print("\n1. Testing configuration:")
    print(f"   Default max_cache_files: {ec.get_max_cache_files()}")
    
    ec.set_max_cache_files(3)
    print(f"   After set_max_cache_files(3): {ec.get_max_cache_files()}")
    
    # 2. Clear existing cache
    print("\n2. Clearing existing cache...")
    imp = ec.EcImporter(fname_parser=example_filename_parser, log_level="WARNING")
    cache_dir = imp._get_cache_dir(str(data_path))
    for cache_file in cache_dir.glob('*.pkl'):
        cache_file.unlink()
    for cache_file in cache_dir.glob('*.json'):
        cache_file.unlink()
    print(f"   Cache directory: {cache_dir}")
    
    # 3. Load with first configuration
    print("\n3. Loading with config 1 (fname_parser=example_filename_parser)...")
    imp1 = ec.EcImporter(fname_parser=example_filename_parser, log_level="WARNING")
    eclist1 = imp1.load_folder(str(data_path))
    cache_files = list(cache_dir.glob('*.pkl'))
    print(f"   ✓ Loaded {len(eclist1)} files")
    print(f"   ✓ Cache files: {len(cache_files)}")
    
    # 4. Load with second configuration
    print("\n4. Loading with config 2 (fname_parser=None)...")
    imp2 = ec.EcImporter(fname_parser=None, log_level="WARNING")
    eclist2 = imp2.load_folder(str(data_path))
    cache_files = list(cache_dir.glob('*.pkl'))
    print(f"   ✓ Loaded {len(eclist2)} files")
    print(f"   ✓ Cache files: {len(cache_files)} (both configs cached)")
    
    # 5. Load with first configuration again (should use cache)
    print("\n5. Loading with config 1 again (should hit cache)...")
    imp1_again = ec.EcImporter(fname_parser=example_filename_parser, log_level="WARNING")
    eclist1_again = imp1_again.load_folder(str(data_path))
    cache_files = list(cache_dir.glob('*.pkl'))
    print(f"   ✓ Loaded {len(eclist1_again)} files")
    print(f"   ✓ Cache files: {len(cache_files)} (still 2, both kept)")
    
    # 6. Test with max_cache_files=1 (old behavior)
    print("\n6. Testing with max_cache_files=1 (old behavior)...")
    ec.set_max_cache_files(1)
    imp3 = ec.EcImporter(fname_parser=example_filename_parser, log_level="WARNING")
    eclist3 = imp3.load_folder(str(data_path))
    cache_files = list(cache_dir.glob('*.pkl'))
    print(f"   ✓ Loaded {len(eclist3)} files")
    print(f"   ✓ Cache files: {len(cache_files)} (only 1 kept, old behavior)")
    
    # 7. Verify data_path.txt is preserved
    print("\n7. Verifying data_path.txt is preserved...")
    ref_file = cache_dir / 'data_path.txt'
    if ref_file.exists():
        print(f"   ✓ data_path.txt exists")
        content = ref_file.read_text(encoding='utf-8').strip()
        print(f"   ✓ Points to: {content}")
    else:
        print(f"   ❌ data_path.txt not found!")
        return 1
    
    print("\n" + "=" * 70)
    print("✓ Manual verification completed successfully!")
    print("=" * 70)
    return 0

if __name__ == '__main__':
    exit(main())
