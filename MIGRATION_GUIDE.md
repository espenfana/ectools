# Migration Guide: Separating Specialized Code from ECTools

This guide explains how to migrate domain-specific auxiliary data sources from the generalized `ectools` framework to your specialized `bcsec` repository.

## Overview

The `ectools` package has been refactored to contain only the generalized framework for electrochemical data analysis. Domain-specific implementations (like furnace loggers, specific potentiostat interfaces, and sample management classes) should be moved to specialized repositories.

## Files to Move

### From ECTools to BCSEC Repository

1. **Auxiliary Data Sources**: Move `bcsec_auxiliary_sources.py` to your `bcsec` repository
   - Contains: `PicoLogger`, `FurnaceLogger`, `JsonSource`, `Oxide`, `OxideSample`
   - These are specialized implementations for your specific equipment and experimental setup

2. **Helper Functions**: Move specialized parsing and utility functions
   - Example: `mc_filename_parser` and other domain-specific parsers

3. **Specialized EC Classes**: Move custom electrochemistry experiment classes
   - Example: `PulsedElectrolysis`, `Electrolysis` classes that extend the base framework

## Updated Import Structure

### In your notebooks (after migration):

```python
# Base framework from generalized ectools
import ectools as ec

# Option 1: Import individual specialized classes
from bcsec.auxiliary_sources import FurnaceLogger, PicoLogger, JsonSource
from bcsec.helper_functions import mc_filename_parser
from bcsec.ec_classes.electrolysis import PulsedElectrolysis, Electrolysis

# Option 2: Use convenience functions for source collections
from bcsec.auxiliary_sources import get_standard_sources, get_all_sources

# Usage with individual imports (same as before)
ec.set_config('cycle_convention', 'init')
imp = ec.EcImporter(
    fname_parser=mc_filename_parser, 
    aux_data_classes=[FurnaceLogger, PicoLogger, JsonSource], 
    log_level='DEBUG'
)

# Usage with convenience functions (simplified)
imp = ec.EcImporter(
    fname_parser=mc_filename_parser,
    aux_data_classes=get_standard_sources(),  # Gets [FurnaceLogger, PicoLogger, JsonSource]
    log_level='DEBUG'
)

# Or use specific source categories
sources = get_all_sources()
imp = ec.EcImporter(
    fname_parser=mc_filename_parser,
    aux_data_classes=sources['temperature'] + sources['electrochemical'],  # Only temp + electrochemical
    log_level='DEBUG'
)

fl = imp.load_folder(CWD, sort_by='starttime')
```

### Available Convenience Functions:

- **`get_standard_sources()`**: Returns `[FurnaceLogger, PicoLogger, JsonSource]` - the typical BCSEC setup
- **`get_temperature_sources()`**: Returns `[FurnaceLogger]` - only temperature monitoring
- **`get_electrochemical_sources()`**: Returns `[PicoLogger]` - only electrochemical monitoring  
- **`get_metadata_sources()`**: Returns `[JsonSource]` - only metadata and sample info
- **`get_all_sources()`**: Returns a dictionary with all categories for flexible selection
- **`STANDARD_SOURCES`**: Pre-defined constant for the standard source list

### In your bcsec repository files:

```python
# Import base framework
import ectools
from ectools.auxiliary_sources import AuxiliaryDataSource
from ectools.classes.electrochemistry import Electrochemistry

# Your specialized implementations
class PicoLogger(AuxiliaryDataSource):
    # ... implementation
```

## What Stays in ECTools

The following components remain in `ectools` as they are part of the generalized framework:

- **Base Classes**: `AuxiliaryDataSource`, `AuxiliaryDataHandler`, `Electrochemistry`
- **Core Framework**: Data loading, interpolation, and visualization infrastructure
- **Configuration System**: Config management and validation
- **Import/Export**: General CSV/data file handling
- **Base Plotting**: Generic matplotlib/bokeh integration

## Migration Steps

1. **Create BCSEC Repository Structure**:
   ```
   bcsec/
   ├── __init__.py
   ├── auxiliary_sources.py      # From bcsec_auxiliary_sources.py
   ├── helper_functions.py       # Your specialized parsers
   └── ec_classes/
       ├── __init__.py
       └── electrolysis.py       # Your specialized EC classes
   ```

2. **Install ECTools as Dependency**: Add `ectools` to your `bcsec` requirements
   ```
   pip install ectools  # or add to requirements.txt
   ```

3. **Update Imports**: Change imports in your notebooks and BCSEC files as shown above

4. **Test Migration**: Verify that all functionality works with the new import structure

## Benefits of This Separation

- **Cleaner Dependencies**: ECTools remains lightweight and general-purpose
- **Easier Maintenance**: Specialized code evolves independently  
- **Better Collaboration**: Others can use ECTools without your specific equipment dependencies
- **Modular Development**: Add new specialized sources without affecting the base framework

## Backward Compatibility

The auxiliary framework in `ectools` maintains full backward compatibility. Your existing notebooks will work unchanged until you migrate the imports. The deprecation warnings provide clear guidance on updating old patterns.

## Support

The auxiliary data framework provides comprehensive logging and error handling to help debug any migration issues. Enable DEBUG logging to see detailed information about data loading and interpolation processes.
