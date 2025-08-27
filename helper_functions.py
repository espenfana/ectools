'''Electrochem# NOTE: This file previously contained specialized helper functions that have been 
# migrated to domain-specific repositories:
#
# - mc_filename_parser: Moved to bcsec.auxiliary_sources 
# - mc_auxiliary_importer: Replaced by new auxiliary framework (AuxiliaryDataHandler)
# - OxideSample: Moved to bcsec.auxiliary_sources
# - display_auxiliary_data: Replaced by auxiliary source visualize() methods
#
# The new auxiliary framework provides better separation of concerns and 
# more robust data handling. See MIGRATION_GUIDE.md for details.
#
# For visualization, use the new auxiliary framework:
#   fl.aux.furnacelogger.visualize()  # Instead of display_auxiliary_data(fl, furnace=True)
#   fl.aux.picologger.visualize()     # Instead of display_auxiliary_data(fl, pico=True)  
#   fl.aux.visualize()                # Visualize all auxiliary sourcested support module

General helper functions for the ectools package.

Note: Domain-specific functions (like mc_filename_parser, mc_auxiliary_importer, 
and display_auxiliary_data) have been moved to specialized repositories.
For BCSEC-specific functions, see bcsec.helper_functions.

'''

import logging

logger = logging.getLogger(__name__)

# NOTE: This file previously contained specialized helper functions that have been 
# migrated to domain-specific repositories:
#
# - mc_filename_parser: Moved to bcsec.helper_functions 
# - mc_auxiliary_importer: Replaced by new auxiliary framework
# - OxideSample: Moved to bcsec.auxiliary_sources
# - display_auxiliary_data: Replaced by auxiliary source visualize() methods
#
# The new auxiliary framework provides better separation of concerns and 
# more robust data handling. See MIGRATION_GUIDE.md for details.

def example_filename_parser(_, fname: str) -> dict:
    """Example filename parser for ectools framework.
    
    This is a template showing how to create custom filename parsers.
    Domain-specific parsers should be implemented in their respective repositories.
    
    Args:
        _: Unused first argument (for compatibility with EcImporter)
        fname: Filename to parse
        
    Returns:
        Dictionary with extracted metadata
        
    Example:
        # Create a custom parser for your filename format
        def my_parser(_, fname: str) -> dict:
            # Extract metadata from filename
            return {'sample_id': 'extracted_value', 'temperature': 25}
            
        # Use with EcImporter
        imp = ec.EcImporter(fname_parser=my_parser)
    """
    # Basic example - extract just the base filename
    import os
    base_name = os.path.splitext(fname)[0]
    
    return {
        'filename': base_name,
        'extension': os.path.splitext(fname)[1]
    }


if __name__ == '__main__':
    # Test the example filename parser
    test_fname = 'example_file.DTA'
    print("Example parser result:", example_filename_parser(None, test_fname))
