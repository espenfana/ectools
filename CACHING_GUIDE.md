# EcTools Caching Guide

## Overview

The caching system in ectools significantly speeds up repeated loads of electrochemistry data by storing parsed `EcList` objects. This is especially valuable when:
- Raw data files are on remote storage (SharePoint, network drives)
- Processing includes expensive operations (auxiliary data, collation)
- You need to repeatedly load the same dataset during analysis

## Quick Start

```python
import ectools as ec

# Create importer
imp = ec.EcImporter(log_level='INFO')

# First load: parses all files and creates cache (~1.6s for 52 files)
fl = imp.load_folder('/path/to/data', use_cache=True)

# Second load: loads from cache (~0.12s for 52 files) - 10-15x faster!
fl = imp.load_folder('/path/to/data', use_cache=True)
```

## Configuration

### Global Configuration

Configure caching behavior for all importers:

```python
import ectools as ec

# Enable/disable caching globally (default: True)
ec.set_cache_enabled(True)
ec.set_cache_enabled(False)

# Check if caching is enabled
enabled = ec.get_cache_enabled()

# Set cache root directory (overrides cache_location)
ec.set_cache_root('C:/code/experimentals')
ec.set_cache_root('/tmp/my_cache')

# Get current cache root
root = ec.get_cache_root()  # Returns None if not set
```

### Per-Importer Configuration

Each importer can have its own cache location:

```python
# Cache in a specific directory
imp = ec.EcImporter(
    cache_root='/tmp/custom_cache',
    log_level='INFO'
)
```

### Cache Location Strategies

The `cache_location` config setting supports multiple strategies:

```python
from ectools import set_config

# Store in data folder itself
set_config('cache_location', 'local')
# Creates: /path/to/data/.ectools_cache/

# Store in project root (default)
set_config('cache_location', 'project')
# Creates: /project/root/.ectools_cache/
# Detects project by looking for: .git, pyproject.toml, setup.py, etc.

# Store in user cache directory
set_config('cache_location', 'user')
# Creates:
#   Windows: %LOCALAPPDATA%\ectools\Cache\.ectools_cache/
#   macOS: ~/Library/Caches/ectools/.ectools_cache/
#   Linux: ~/.cache/ectools/.ectools_cache/

# Use absolute path
set_config('cache_location', '/custom/path')
```

## Cache Directory Structure

```
{cache_root}/.ectools_cache/{path_hash}/
├── {cache_key}.pkl          # Pickled EcList
├── {cache_key}.json         # Metadata (timestamp, file count, size)
└── data_path.txt            # Reference to original data path
```

- `{path_hash}`: First 12 characters of MD5 hash of data folder path
- `{cache_key}`: MD5 hash of all files + configuration

## Usage Examples

### Example 1: SharePoint Data with Local Cache

```python
import ectools as ec

# Configure cache to store locally
ec.set_cache_root('C:/code/experimentals')

imp = ec.EcImporter(log_level='INFO')

# Data on SharePoint
sharepoint_path = '//sharepoint/site/data/experiment_2024'

# First load: slow (reads from SharePoint)
fl = imp.load_folder(sharepoint_path, use_cache=True)

# Subsequent loads: fast (from local cache)
fl = imp.load_folder(sharepoint_path, use_cache=True)
```

### Example 2: Different Sorting (Reuses Cache)

```python
# Load unsorted
fl1 = imp.load_folder(data_path, use_cache=True)

# Load sorted by starttime (still fast - uses same cache!)
fl2 = imp.load_folder(data_path, use_cache=True, sort_by='starttime')

# Load sorted by fname (still fast!)
fl3 = imp.load_folder(data_path, use_cache=True, sort_by='fname')
```

### Example 3: Disabling Cache Temporarily

```python
# Load without using or creating cache
fl = imp.load_folder(data_path, use_cache=False)
```

### Example 4: Disabling Cache Globally

```python
# Disable caching for all operations
ec.set_cache_enabled(False)

# This won't use or create cache
fl = imp.load_folder(data_path)

# Re-enable
ec.set_cache_enabled(True)
```

## Cache Management

### Get Cache Information

```python
# Get detailed cache info
info = imp.cache_info(data_path)

print(f"Cache exists: {info['cache_exists']}")
print(f"Cache directory: {info['cache_dir']}")
print(f"Original data path: {info['data_path']}")
print(f"Number of cache files: {info['n_cache_files']}")
print(f"Total size: {info['total_size_mb']:.1f} MB")

# Detailed info for each cache file
for cache_file in info['cache_files']:
    print(f"  Key: {cache_file['cache_key']}")
    print(f"  Size: {cache_file['size_mb']:.1f} MB")
    print(f"  Files: {cache_file['file_count']}")
```

### Clear Cache

```python
# Clear cache for specific folder
imp.clear_cache(data_path)

# Clear all caches
imp.clear_cache(all_caches=True)

# Alternative: clear with no path specified
imp.clear_cache()  # Also clears all caches
```

## Cache Invalidation

The cache is **automatically invalidated** when:

### ✅ Any file changes
- File modified (different mtime or size)
- File added to folder
- File removed from folder
- Auxiliary file changes

### ✅ Configuration changes
- Different `collation_mapping`
- Different `aux_data_classes`
- Different `fname_parser`
- Different `data_folder_id` or `aux_folder_id`

### ❌ NOT invalidated when
- `sort_by` parameter changes (by design - sorting happens after cache load)

## Advanced Usage

### Custom Filename Parser

```python
def my_parser(fpath, fname):
    # Parse custom metadata
    return {'custom_field': 'value'}

imp = ec.EcImporter(
    fname_parser=my_parser,
    cache_root='/tmp/cache'
)

# Cache key includes parser function name
# Changing parser invalidates cache
fl = imp.load_folder(data_path, use_cache=True)
```

### With Auxiliary Data

```python
from my_aux import TemperatureData, PicoData

imp = ec.EcImporter(
    aux_data_classes=[TemperatureData, PicoData],
    cache_root='/tmp/cache'
)

# Cache key includes auxiliary class names
# Changing aux classes invalidates cache
fl = imp.load_folder(data_path, use_cache=True)
```

### With Collation Mapping

```python
from my_classes import PulsedElectrolysis

collation_config = {
    PulsedElectrolysis: {
        'id_numbers': [4, 9],
        'cyclic': False
    }
}

# Cache key includes collation config
# Changing collation invalidates cache
fl = imp.load_folder(
    data_path, 
    collation_mapping=collation_config,
    use_cache=True
)
```

## Performance Characteristics

### Cache Save
- Time: ~1-5 seconds for typical dataset
- Overhead: ~5-10% of normal load time
- Size: ~10-50 MB per 10 files (depends on data)

### Cache Load
- Time: <1 second for any size dataset
- Speedup: 10-100x faster for loads with auxiliary sources
- Memory: Same as normal load

### Cache Storage
- Format: Python pickle (HIGHEST_PROTOCOL)
- Compression: None (but pickle is efficient)
- Metadata: JSON file alongside pickle

## Best Practices

1. **Use project-level cache for development**
   ```python
   # Automatic with default settings
   imp = ec.EcImporter()
   ```

2. **Use custom cache for production**
   ```python
   ec.set_cache_root('/data/cache')
   ```

3. **Disable cache for debugging**
   ```python
   fl = imp.load_folder(path, use_cache=False)
   ```

4. **Clear cache after major changes**
   ```python
   imp.clear_cache(all_caches=True)
   ```

5. **Check cache size periodically**
   ```python
   info = imp.cache_info(path)
   if info['total_size_mb'] > 1000:  # 1 GB
       imp.clear_cache(path)
   ```

## Troubleshooting

### Cache not working?

1. Check if caching is enabled:
   ```python
   print(ec.get_cache_enabled())
   ```

2. Check cache location:
   ```python
   info = imp.cache_info(data_path)
   print(info['cache_dir'])
   ```

3. Check for permissions:
   ```python
   import os
   cache_dir = info['cache_dir']
   print(f"Writable: {os.access(cache_dir, os.W_OK)}")
   ```

### Cache keeps invalidating?

The cache invalidates when files change. Check:
- Are files being modified during load?
- Are there background processes touching files?
- Use `cache_info()` to inspect cache key

### Cache taking too much space?

```python
# Clear old caches
imp.clear_cache(all_caches=True)

# Or set cache location to temp directory
ec.set_cache_root('/tmp/ectools_cache')
```

## Security Considerations

- Cache files contain pickled Python objects
- Only load caches created by trusted code
- Use project-specific cache directories to avoid conflicts
- Cache files are NOT encrypted

## Version Compatibility

- Cache format: Python pickle (HIGHEST_PROTOCOL)
- Compatible across Python versions that support the pickle protocol
- Cache invalidates automatically if data or config changes
- Consider clearing cache after major ectools upgrades

## Summary

The caching system provides:
- ✅ Automatic cache invalidation
- ✅ Configurable storage locations  
- ✅ Minimal overhead
- ✅ Significant speedup (10-100x)
- ✅ Easy cache management
- ✅ No code changes required for basic use

Simply add `use_cache=True` to `load_folder()` calls to enable caching!
