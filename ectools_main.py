'''ectools_main.py'''
# Standard library imports
import logging
import os
import re
import pickle
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd


# Relational imports
from .classes import EcList, ElectroChemistry, ElectrochemicalImpedance
from .config import get_config, get_post_process, get_cache_enabled, get_cache_root
from .auxiliary_sources import AuxiliaryDataHandler
import warnings
# classes is a collection of container objects meant for different methods

class EcImporter:
    """
    EcImporter is a class for importing and parsing electrochemistry files from a specified folder.
    Attributes:
        fname_parser (function): Optional function to parse information from the file name and path.
        log_level (str): Logging level for the logger.
        logger (logging.Logger): Logger instance for logging messages and errors.
    Methods:
        load_folder(fpath: str, **kwargs) -> EcList:
            Parse and load the contents of a folder (not subfolders).
        load_file(fpath: str, fname: str):
            Load and parse an electrochemistry file.
    """
    def __init__(self, fname_parser=None, aux_data_classes=None, log_level="WARNING", 
                 cache_root=None, **kwargs):
        '''
        fname_parser: optional function to parse information from the file name and path.
            Expected to return a dictionary, from which the key-value pairs are added to the 
            container object returned from load_file.
        log_level: logging level for the logger as a string (e.g., "DEBUG", "INFO", "ERROR").
        cache_root: optional cache root directory override for this importer instance.
        kwargs: key-val pairs added to this instance.
        '''
        self.fname_parser = fname_parser
        #self.aux_importer = aux_importer
        self.aux_data_classes = aux_data_classes or []
        self.cache_root = cache_root
        self._setup_logging(log_level)
 
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup_logging(self, log_level):
        '''Set up logging for the EcImporter class.'''
        # Set the root ectools logger level so all children inherit it
        root_logger = logging.getLogger('ectools')
        
        # Map string log levels to logging module levels
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        root_logger.setLevel(log_levels.get(log_level.upper(), logging.ERROR))
        
        # Set up console handler if not already present
        if not root_logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_levels.get(log_level.upper(), logging.ERROR))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Your existing logger setup for the importer itself
        self.logger = logging.getLogger(__name__)

    def _setup_logging_old(self, log_level):
        '''Set up logging for the EcImporter class.'''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set logger to capture all levels, handlers will filter

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create handlers
        console_handler = logging.StreamHandler()

        # Map string log levels to logging module levels
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        # Set logging levels
        console_handler.setLevel(log_levels.get(log_level.upper(), logging.ERROR))

        # Create formatters and add them to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)

    def _find_project_root(self, start_path):
        """
        Find project root directory by looking for common project markers.
        
        Args:
            start_path: Path to start searching from
            
        Returns:
            Path: Project root directory or start_path if not found
        """
        markers = ['.git', 'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt', '.gitignore']
        current = Path(start_path).resolve()
        
        # Walk up the directory tree
        for parent in [current] + list(current.parents):
            for marker in markers:
                if (parent / marker).exists():
                    self.logger.debug('Found project root at %s (marker: %s)', parent, marker)
                    return parent
        
        self.logger.warning('No project root found, using start path: %s', start_path)
        return current

    def _get_cache_dir(self, data_path):
        """
        Determine cache directory based on configuration.
        
        Args:
            data_path: Path to the data folder being cached
            
        Returns:
            Path: Cache directory for this data folder
        """
        # Determine cache root
        cache_root = None
        
        # Check instance cache_root first
        if self.cache_root:
            cache_root = Path(self.cache_root)
        # Then check config cache_root
        elif get_cache_root():
            cache_root = Path(get_cache_root())
        # Then use cache_location setting
        else:
            cache_location = get_config('cache_location') or 'project'
            
            if cache_location == 'local':
                # Store in data folder itself
                cache_root = Path(data_path)
            elif cache_location == 'project':
                # Find project root and use it
                cache_root = self._find_project_root(data_path)
            elif cache_location == 'user':
                # Use user cache directory (platform-specific)
                if os.name == 'nt':  # Windows
                    cache_root = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'ectools' / 'Cache'
                elif os.sys.platform == 'darwin':  # macOS
                    cache_root = Path.home() / 'Library' / 'Caches' / 'ectools'
                else:  # Linux/Unix
                    cache_root = Path.home() / '.cache' / 'ectools'
            else:
                # Assume it's an absolute path
                cache_root = Path(cache_location)
        
        # Create hash of data path for subdirectory
        data_path_resolved = str(Path(data_path).resolve())
        path_hash = hashlib.md5(data_path_resolved.encode()).hexdigest()[:12]
        
        # Create cache directory structure
        cache_dir = cache_root / '.ectools_cache' / path_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Write reference file
        ref_file = cache_dir / 'data_path.txt'
        if not ref_file.exists():
            try:
                ref_file.write_text(data_path_resolved, encoding='utf-8')
            except Exception as e:
                self.logger.warning('Could not write cache reference file: %s', e)
        
        self.logger.debug('Cache directory: %s', cache_dir)
        return cache_dir

    def _get_cache_key(self, fpath, collation_mapping, aux_data_classes, 
                       data_folder_id, aux_folder_id, fname_parser):
        """
        Generate cache key based on all files in the folder and configuration.
        
        Args:
            fpath: Root folder path
            collation_mapping: Collation configuration
            aux_data_classes: Auxiliary data classes
            data_folder_id: Data folder identifier
            aux_folder_id: Auxiliary folder identifier
            fname_parser: Filename parser function
            
        Returns:
            str: MD5 hash representing this specific dataset and configuration
        """
        cache_data = {
            'files': [],
            'config': {}
        }
        
        # Gather ALL files (respecting folder filtering logic from load_folder)
        data_id = data_folder_id or get_config('data_folder_identifier') or 'data'
        
        for root, dirs, files in os.walk(fpath):
            folder_name = os.path.basename(root)
            
            # Apply same filtering logic as load_folder for data folders
            if root != fpath:
                if folder_name.startswith('_'):
                    continue
                if data_id.lower() not in folder_name.lower():
                    continue
            
            # Include all files in valid folders
            for fname in files:
                file_path = os.path.join(root, fname)
                try:
                    stat = os.stat(file_path)
                    rel_path = os.path.relpath(file_path, fpath)
                    cache_data['files'].append({
                        'path': rel_path,
                        'mtime': stat.st_mtime,
                        'size': stat.st_size
                    })
                except (OSError, PermissionError) as e:
                    self.logger.debug('Could not stat file %s: %s', file_path, e)
        
        # Add configuration that affects loaded data
        cache_data['config']['collation_mapping'] = str(collation_mapping) if collation_mapping else None
        cache_data['config']['aux_data_classes'] = [cls.__name__ for cls in (aux_data_classes or [])]
        cache_data['config']['fname_parser'] = fname_parser.__name__ if fname_parser else None
        cache_data['config']['data_folder_id'] = data_id
        cache_data['config']['aux_folder_id'] = aux_folder_id or get_config('auxiliary_folder_identifier') or 'aux'
        # Explicitly exclude sort_by - it doesn't affect data, only order
        
        # Generate hash
        cache_str = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        
        self.logger.debug('Cache key generated: %s (based on %d total files)', 
                         cache_hash, len(cache_data['files']))
        
        return cache_hash

    def _save_cache(self, cache_file, eclist):
        """
        Save EcList to cache file.
        
        Args:
            cache_file: Path to cache file
            eclist: EcList object to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Save pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(eclist, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata_file = cache_file.with_suffix('.json')
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'file_count': len(eclist),
                'cache_size_bytes': os.path.getsize(cache_file)
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            size_mb = metadata['cache_size_bytes'] / (1024 * 1024)
            self.logger.info('Cache saved successfully: %d files, %.2f MB', 
                           metadata['file_count'], size_mb)
            return True
            
        except Exception as e:
            self.logger.warning('Failed to save cache: %s', e)
            # Clean up partial files
            try:
                if cache_file.exists():
                    cache_file.unlink()
                metadata_file = cache_file.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()
            except Exception:
                pass
            return False

    def _load_cache(self, cache_file):
        """
        Load EcList from cache file.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            EcList or None: Loaded EcList or None if failed
        """
        try:
            with open(cache_file, 'rb') as f:
                eclist = pickle.load(f)
            
            self.logger.info('Cache loaded successfully: %d files', len(eclist))
            return eclist
            
        except Exception as e:
            self.logger.warning('Failed to load cache: %s', e)
            return None

    def cache_info(self, fpath):
        """
        Get information about cached data for a folder.
        
        Args:
            fpath: Path to data folder
            
        Returns:
            dict: Cache information
        """
        try:
            cache_dir = self._get_cache_dir(fpath)
            
            # Read data path reference
            ref_file = cache_dir / 'data_path.txt'
            data_path = ref_file.read_text(encoding='utf-8').strip() if ref_file.exists() else 'Unknown'
            
            # Find all cache files
            cache_files = list(cache_dir.glob('*.pkl'))
            
            cache_info_list = []
            total_size = 0
            
            for cache_file in cache_files:
                metadata_file = cache_file.with_suffix('.json')
                
                file_info = {
                    'cache_key': cache_file.stem,
                    'cache_file': str(cache_file),
                    'size_mb': os.path.getsize(cache_file) / (1024 * 1024)
                }
                total_size += os.path.getsize(cache_file)
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        file_info.update(metadata)
                    except Exception:
                        pass
                
                cache_info_list.append(file_info)
            
            return {
                'cache_exists': len(cache_files) > 0,
                'cache_dir': str(cache_dir),
                'data_path': data_path,
                'n_cache_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_files': cache_info_list
            }
            
        except Exception as e:
            self.logger.error('Error getting cache info: %s', e)
            return {
                'cache_exists': False,
                'error': str(e)
            }

    def clear_cache(self, fpath=None, all_caches=False):
        """
        Clear cache files.
        
        Args:
            fpath: Path to data folder (if None and all_caches=False, clears all)
            all_caches: If True, clear all caches in cache root
            
        Returns:
            bool: True if successful
        """
        try:
            if all_caches or fpath is None:
                # Clear entire .ectools_cache directory
                cache_root = None
                
                if self.cache_root:
                    cache_root = Path(self.cache_root)
                elif get_cache_root():
                    cache_root = Path(get_cache_root())
                else:
                    cache_location = get_config('cache_location') or 'project'
                    if cache_location == 'project':
                        # Use current working directory as fallback
                        cache_root = self._find_project_root(os.getcwd())
                    elif cache_location == 'user':
                        if os.name == 'nt':
                            cache_root = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'ectools' / 'Cache'
                        elif os.sys.platform == 'darwin':
                            cache_root = Path.home() / 'Library' / 'Caches' / 'ectools'
                        else:
                            cache_root = Path.home() / '.cache' / 'ectools'
                    else:
                        cache_root = Path(cache_location)
                
                cache_dir = cache_root / '.ectools_cache'
                
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    self.logger.info('Cleared all caches in: %s', cache_dir)
                else:
                    self.logger.info('No cache directory found at: %s', cache_dir)
                
            else:
                # Clear cache for specific folder
                cache_dir = self._get_cache_dir(fpath)
                
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    self.logger.info('Cleared cache for: %s', fpath)
                else:
                    self.logger.info('No cache found for: %s', fpath)
            
            return True
            
        except Exception as e:
            self.logger.error('Error clearing cache: %s', e)
            return False

    def load_folder(self, fpath: str, data_folder_id=None, aux_folder_id=None, sort_by=None, 
                    collation_mapping=None, aux_only=False, use_cache=None, **kwargs) -> EcList:
        '''
        Parse and load the contents of a folder and its subfolders.
        Ignores folders starting with '_' and only includes subfolders containing data_folder_id.
        
        Args:
            fpath: Root folder path to process
            data_folder_id: Override for data folder identifier (default from config)
            aux_folder_id: Override for aux folder identifier (default from config)
            sort_by: Attribute name to sort the EcList by (e.g., 'starttime', 'fname'). 
                    Default None (no sorting, files in order read)
            collation_mapping: Dict mapping target classes to configurations for collating files.
                    Supports two formats:
                    1. Simple format: {target_class: {'id_numbers': [list], 'cyclic': bool, 'kwargs': dict}}
                    2. Enhanced format: {target_class: {'id_numbers': {id: kwargs_dict}, 'cyclic': bool}}
                    Examples:
                    Simple: {PulsedElectrolysis: {'id_numbers': [4, 9], 'cyclic': False, 
                             'kwargs': {'default_label': 'experiment'}}}
                    Enhanced: {PulsedElectrolysis: {'id_numbers': {4: {'label': 'E1 experiment'}, 
                               9: {'label': 'E3 experiment'}}, 'cyclic': False}}
            aux_only: If True, skip loading electrochemistry data files and only load auxiliary data.
                     Useful for quick access to temperature, pico data, etc. Default False.
            use_cache: If True, use cache; if False, skip cache; if None, use config default.
            **kwargs: Additional arguments passed to EcList
        '''
        # Determine if caching enabled
        if use_cache is None:
            use_cache = get_cache_enabled()
        
        # Get folder identifiers from config or use overrides
        data_id = data_folder_id or get_config('data_folder_identifier') or 'data'
        aux_id = aux_folder_id or get_config('auxiliary_folder_identifier') or 'aux'
        
        # Try to load from cache
        if use_cache:
            try:
                cache_dir = self._get_cache_dir(fpath)
                cache_key = self._get_cache_key(fpath, collation_mapping, self.aux_data_classes,
                                               data_folder_id, aux_folder_id, self.fname_parser)
                cache_file = cache_dir / f"{cache_key}.pkl"
                
                if cache_file.exists():
                    self.logger.info('Cache hit - loading from cache')
                    eclist = self._load_cache(cache_file)
                    if eclist is not None:
                        # Apply sorting AFTER loading from cache
                        # (so different sort_by doesn't create new cache)
                        if sort_by and len(eclist) > 0:
                            try:
                                if hasattr(eclist[0], sort_by):
                                    eclist.sort(key=lambda f: getattr(f, sort_by))
                                    self.logger.info('Sorted EcList by attribute: %s', sort_by)
                                else:
                                    self.logger.warning('Sort attribute "%s" not found in files, skipping sort', sort_by)
                            except Exception as e:
                                self.logger.warning('Error sorting by "%s": %s', sort_by, e)
                        return eclist
                else:
                    self.logger.info('Cache miss - loading data from files')
            except Exception as e:
                self.logger.warning('Error checking cache: %s. Loading from files.', e)
        
        self.logger.debug('Using folder identifiers - data: "%s", aux: "%s"', data_id, aux_id)
        
        eclist = EcList(fpath=fpath, **kwargs)
        
        if aux_only:
            self.logger.info('aux_only=True: Skipping electrochemistry data file loading')
        else:
            all_files = []
            
            for root, dirs, files in os.walk(fpath):
                folder_name = os.path.basename(root)
                
                # Always include main folder, skip subfolders starting with '_'
                if root != fpath:
                    if folder_name.startswith('_'):
                        continue
                    # Only include subfolders containing data_id (case-insensitive)
                    if data_id.lower() not in folder_name.lower():
                        continue
                        
                try:
                    all_files.extend([(root, f) for f in files if os.path.isfile(os.path.join(root, f))])
                    if root != fpath:
                        self.logger.debug('Included subfolder: %s', root)
                except (OSError, PermissionError) as e:
                    self.logger.warning('Error accessing folder %s: %s', root, e)
            
            self.logger.info('Found %d files to process', len(all_files))
            ignored = 0
            
            # Log progress at regular intervals
            progress_interval = max(1, len(all_files) // 10)  # Log every 10% or at least every file
            
            for i, (file_path, fname) in enumerate(all_files):
                try:
                    f = self.load_file(file_path, fname)
                    if f:
                        eclist.append(f)
                    else:
                        ignored += 1
                except (FileNotFoundError, PermissionError, UnicodeDecodeError, RuntimeError) as e:
                    self.logger.warning('Error processing file %s: %s', fname, e)
                    ignored += 1
                
                # Log progress at intervals
                if (i + 1) % progress_interval == 0 or i == len(all_files) - 1:
                    self.logger.info('Processed %d/%d files (%d parsed, %d ignored)', 
                                   i + 1, len(all_files), len(eclist), ignored)
            
            self.logger.info('Completed processing: %d files total, %d parsed, %d ignored', 
                            len(all_files), len(eclist), ignored)
        
        # Process collation mapping before auxiliary data handling
        if collation_mapping:
            self.logger.info('Processing collation mapping for %d target classes...', len(collation_mapping))
            eclist = self._process_collation_mapping(eclist, collation_mapping)
        
        # New logic for aux importing
        if self.aux_data_classes:
            self.logger.info('Importing auxiliary data...')
            eclist.aux = AuxiliaryDataHandler(main_path=fpath, aux_data_classes=self.aux_data_classes)
            eclist.aux.import_auxiliary_data()
            
            # Count successful vs failed sources
            total_requested = len(self.aux_data_classes)
            successful_sources = len(eclist.aux.sources)
            failed_sources = total_requested - successful_sources
            
            if successful_sources > 0:
                # Add interpolated and calculated data axes
                interpolated_count = 0
                skipped_count = 0
                for f in eclist:
                    # Skip techniques that don't have standard time-series potential data
                    if isinstance(f, ElectrochemicalImpedance):
                        skipped_count += 1
                        self.logger.debug('Skipping auxiliary data interpolation for EIS file: %s', f.fname)
                        continue
                    
                    for source_name in eclist.aux.sources:
                        source = getattr(eclist.aux, source_name, None)  # More explicit None default
                        if source is not None and hasattr(source, 'continuous_data') and source.continuous_data:
                            try:
                                new_columns = source.interpolate_data_columns(f.timestamp, f.pot)
                                if new_columns:  # Only process if we got data back
                                    for column_name, data_array in new_columns.items():
                                        display_name = source.main_data_columns[column_name]
                                        setattr(f, column_name, data_array)
                                        f.data_columns[column_name] = display_name
                                    interpolated_count += len(new_columns)
                                    self.logger.debug('Interpolated %d columns from %s for file %s', 
                                                    len(new_columns), source_name, f.fname)
                            except Exception as e:
                                self.logger.warning('Error interpolating data from %s for file %s: %s', 
                                                  source_name, f.fname, e)
                
                # More accurate reporting
                log_msg = f'Auxiliary data processing completed: {successful_sources}'
                if failed_sources > 0:
                    log_msg += f'/{total_requested} sources successful'
                else:
                    log_msg += ' sources'
                log_msg += f', {interpolated_count} interpolated columns total'
                if skipped_count > 0:
                    log_msg += f' ({skipped_count} files skipped - incompatible technique)'
                self.logger.info(log_msg)
            else:
                if failed_sources > 0:
                    self.logger.warning('All %d auxiliary data sources failed to load', failed_sources)
                else:
                    self.logger.info('No auxiliary data sources configured')
        # Apply post-processing special logic if available
        POST_PROCESS = get_post_process()
        self.logger.debug(f'Post-processing with special logic: {POST_PROCESS is not None}')
        if POST_PROCESS:
            if callable(POST_PROCESS):
                self.logger.info('Applying post-processing special logic...')
                try:
                    POST_PROCESS(eclist)
                    self.logger.info('Post-processing completed successfully')
                except Exception as e:
                    self.logger.error('Error during post-processing: %s', e)
            else:
                self.logger.warning('POST_PROCESS is not callable, skipping post-processing')
        # if self.aux_importer:
        #     try:
        #         self.logger.info('Importing auxiliary data...')
        #         eclist.aux = self.aux_importer(fpath, aux_folder_id=aux_id)
        #     except RuntimeError as e:
        #         self.logger.warning('Error importing auxiliary data: %s', e)
        #         eclist.aux = None
        #     if eclist.aux is not None:
        #         try: # Attempt to associate auxiliary data with each file
        #             for f in eclist:
        #                 f.aux = self._associate_auxiliary_data(eclist.aux, f)
        #             self.logger.info('Successfully associated auxiliary data with %d files', len(eclist))
        #         except (KeyError, ValueError, TypeError, AttributeError) as e:
        #             self.logger.exception('Error associating auxiliary data: %s', e)
        
        # Sort the EcList if sort_by parameter is provided
        # Note: This happens after collation, so collated objects will be sorted among themselves
        # and with any remaining individual files
        if sort_by and len(eclist) > 0:
            try:
                # Check if the first item has the requested attribute
                if hasattr(eclist[0], sort_by):
                    eclist.sort(key=lambda f: getattr(f, sort_by))
                    self.logger.info('Sorted EcList by attribute: %s', sort_by)
                else:
                    self.logger.warning('Sort attribute "%s" not found in files, skipping sort', sort_by)
            except Exception as e:
                self.logger.warning('Error sorting by "%s": %s', sort_by, e)
        
        eclist._generate_fid_idx()  # pylint: disable=protected-access #(because timing)
        
        # Save to cache if enabled and not already loaded from cache
        if use_cache:
            try:
                self._save_cache(cache_file, eclist)
            except Exception as e:
                self.logger.warning('Error saving cache: %s', e)
        
        return eclist

    def load_file(self, fpath, fname):
        '''Load and parse an electrochemistry file. Opens the file to look for an identifier, 
        then passes it along to the correct file parser.
        '''
        try:
            with open(os.path.join(fpath, fname), encoding='utf8') as f:
                row_1 = f.readline().strip()
                if re.match('EC-Lab ASCII FILE', row_1):  # File identified as EC-lab
                    container = self._parse_file_mpt(fname, fpath)
                elif re.match('EXPLAIN', row_1):
                    container = self._parse_file_gamry(fname, fpath)
                else:
                    self.logger.info('Not a recognized format, skipping: %s', fname)
                    return None
                if self.fname_parser:
                    fname_dict = self.fname_parser(fpath, fname)
                    for key, val in fname_dict.items():
                        setattr(container, key, val)
                container.label = getattr(container, 'id_full', fname)
                self.logger.info('Successful import: %s', fname)
                return container
        except (IOError, OSError, FileNotFoundError, UnicodeDecodeError):
            # File cannot be opened, log and skip
            self.logger.info('File cannot be opened, skipping: %s', fname)
            return None
        except Exception as error:  # pylint: disable=broad-except
            self.logger.error('Error loading file %s: %s', fname, error, exc_info=True)
            raise error
    
    # def collate_convert(self, source, target_class):
    #     '''Rebuild the source file or files into a new containter class'''

    
    # filename, data_dict, aux_dict, meta_dict 

    def _parse_file_gamry(self, fname, fpath):
        '''Parse a Gamry formatted ascii file (such as .-DAT). 
        Returns a custom electrochemistry container object '''
        meta_list = []
        try:
            with open(os.path.join(fpath, fname), encoding="utf-8") as f:
                # Open the file to read the first lines
                technique = None
                ocv_delay_time = 0
                while line := f.readline():
                    if line == "": # at EOF, readline() will return an empty string
                        raise RuntimeError('No TABLE detected')
                    line = line.rstrip().split('\t')
                    if 'TABLE' in line:
                        if 'OCVCURVE' in line: # TODO add saving of OCP curve
                            num_lines = int(line[-1]) + 2 # 2 header lines
                            ocv_lines = [f.readline() for _ in range(num_lines)]
                            last_line = ocv_lines[-1].strip().split('\t')
                            self.logger.debug(last_line)
                            ocv_delay_time = float(last_line[1])
                        else:
                            break
                    meta_list.append(line)
                    if 'TAG' in line:
                        technique = line[1]

                # With the technique from the metadata, we can use the appropriate container object
                container_class = self._get_class(technique)
                container = container_class(fname, fpath, meta_list)
                container.ocv_delay_time = ocv_delay_time
# -- start of data block parsing

                #next two lines should be header and units
                header_row = f.readline().split('\t')
                units_row = f.readline().split('\t')
                coln = {} # identifier to column number dictionary
                units = {} # identifier to column unit dictionary
                for i, column_header in enumerate(header_row):
                    for key, id_tuple in container._column_patterns.items():
                        for id_rgx in id_tuple:
                            if re.match(id_rgx, column_header):
                                coln[key] = i
                                units[key] = units_row[i]

                data_block = []
                if technique == 'CV':
                    # gamry CV's require custom handing due to "CURVE"s being separate tables
                    cycle_list_v2 = []
                    ncycle = 1
                    while line := f.readline():
                        if line[:5] == 'CURVE':
                            ncycle +=1
                            assert f.readline().split('\t') == header_row
                            assert f.readline().split('\t') == units_row
                            continue
                        if 'EXPERIMENTABORTED' in line:
                            break
                        data_block.append(line.split('\t'))
                        cycle_list_v2.append(ncycle)
                else:  # assuming other types have a single data table
                    while line := f.readline():
                        if 'EXPERIMENTABORTED' in line:
                            break
                        data_block.append(line.split('\t'))
                
                if len(data_block) == 0: # Consider filtering out files with empty data blocks
                    self.logger.warning('Data block is empty for file: %s', fname)
                # Reformat the data block into numpy arrays
                for key, j in coln.items():
                    container[key] = np.array(
                    [float(line[j]) if line[j] else np.nan for line in data_block],
                    dtype='float'
                )
                if technique == 'CV':
                    # Add column for sweep direction and cycle number
                    sweep_dir = np.ones_like(container['pot'])
                    if len(container['pot']) > 1:
                        # Determine the sweep direction for the first point
                        sweep_dir[0] = np.where(container['pot'][1] < container['pot'][0], -1, 1)
                    # Use np.where for the rest of the points
                    sweep_dir[1:] = np.where(container['pot'][1:] < container['pot'][:-1], -1, 1)
                    container['sweep_dir'] = sweep_dir
                    vertex_indices = np.where(np.diff(sweep_dir) != 0)[0] + 1  # +1 to adjust index after diff
                    vertex_count = len(vertex_indices)
                    # Log the total count and indices of vertices
                    self.logger.debug('Total vertex count: %d, indices %s', vertex_count, vertex_indices.tolist())
                    # Create column for cycle number (init convention)
                    init_sweep_dir = sweep_dir[0]
                    init_pot = container['pot'][0]

                    # Compute adjusted potential differences
                    delta_pot = (container['pot'] - init_pot) * init_sweep_dir

                    # Find indices where delta_pot crosses zero from negative to positive
                    crossings = np.where((delta_pot[:-1] < 0) & (delta_pot[1:] >= 0))[0] + 1

                    # Initialize cycle increments array
                    cycle_increments = np.zeros_like(container['pot'], dtype=int)
                    cycle_increments[crossings] = 1
                    cycle_increments[0] = 1  # Start with cycle number 1

                    # Compute cycle numbers by cumulative sum
                    cycle_numbers = np.cumsum(cycle_increments)

                    # Assign cycle numbers to the container
                    container['cycle_init'] = cycle_numbers
                    container['cycle_v2'] = np.array(cycle_list_v2, dtype='int')
                    self.logger.debug('Cycle numbers: %s', cycle_numbers)
                    cycle_convension = get_config('cycle_convension')
                    if cycle_convension == 'v2':
                        container['cycle'] = container['cycle_v2']
                    elif cycle_convension == 'init':    
                        container['cycle'] = container['cycle_init']
                    else:
                        raise ValueError(f'Invalid cycle convention: {cycle_convension}')
            container.units = units
            container.parse_meta_gamry()

            if np.any(container.curr):
                container.curr_dens = np.divide(container.curr, container.area)
            return container
        except Exception as error:  # pylint: disable=broad-except
            raise RuntimeError('Error parsing Gamry file') from error

    def _parse_file_mpt(self, fname, fpath): # Untested in new ectools
        '''
        Not tested in new ectools
        Parse an EC-lab ascii file. 
        Returns a custom electrochemistry container object container object'''
        try:
            meta_list = []
            with open(os.path.join(fpath, fname), encoding="utf-8") as f:
                # Open the file to read the first lines
                meta_list.append(f.readline().strip()) # EC-Lab
                meta_list.append(f.readline().strip()) # Contains the number of lines in the meta block
                head_row = int(re.findall(r'\d\d', meta_list[1])[0]) -1 # Header row
                for _ in range(2, head_row+1):
                    meta_list.append(f.readline().strip()) # Read the rest of the metadata block

            # Grabbing the technique from the metadata, we can use the appropriate container object
            technique = meta_list[3] # The fourth line of the block should contain the EC-Lab technique
            container_class = self._get_class(technique) # Match the technique to the container classes
            container = container_class(fname, fpath, meta_list) # Initialize the container
            headers = meta_list[-1].split('\t') # Isolate header row and split the string

            coln = {}
            units = {}
            # Match the columns the container class expects with the columns in the header.
            # Need to maintain order as usecols does not!
            for i, colh in enumerate(headers):
                for key, id_tuple in container._column_patterns.items():
                    for id_rgx in id_tuple:
                        m = re.match(id_rgx, colh)
                        if m:
                            coln[key] = i
                            if m.groups():
                                units[key] = m.groups()[-1]
                            continue

            # Using pandas.read_csv because it is faster than any other csv importer
            # methods i've tried AND it interprets data types
            df = pd.read_csv(fpath + fname,
                encoding='ANSI',
                sep='\t',
                header=head_row,
                skip_blank_lines=False, # ensures the head_row is correct
                index_col=False,
                usecols=coln.values() # ! order is not maintained
                )
            df.columns = coln.keys()
            for key in df.columns:
                container[key] = df[key].to_numpy()
            del df
            container.units = units
            container.parse_meta_mpt()

            return container
        except Exception as error:# pylint: disable=broad-except
            error_message = f'-F- {fpath}{fname}\necImporter._parse_file_mpt error\n{error}'
            self.logger.error(error_message)
            raise error

    def _get_class(self, ident: str) -> ElectroChemistry:
        '''
        Tries to match the identifier with the container classes available.
        
        Parameters:
            ident (str): The identifier string to match with container classes.
        
        Returns:
            ElectroChemistry: The matched container class or the default ElectroChemistry class.
        '''
        if ident is None:
            return ElectroChemistry
        for child in ElectroChemistry.__subclasses__():
            for class_ident in child._identifiers:
                if re.match(class_ident, ident):
                    return child
        return ElectroChemistry  # If no identifier is matched, return ElectroChemistry class

    def _associate_auxiliary_data(self, aux, f) -> Dict: # DEPRICATED
        '''Associate aux data with a file time span'''
        # Convert start and end timestamps
        warnings.warn(
            "The _associate_auxiliary_data method is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2
        )
        tstart = np.datetime64(f.timestamp[0])
        tend = np.datetime64(f.timestamp[-1])

        # Initialize the faux dictionary with empty sub-dictionaries
        faux = {'pico': {}, 'furnace': {}}

        # Process Pico Data if Available
        if aux.get('pico'):
            pico_data = aux['pico']
            if 'timestamp' in pico_data and pico_data['timestamp'] is not None:
                try:
                    # Handle timezone-aware timestamps properly
                    if hasattr(f.timestamp, 'dt') and f.timestamp.dt.tz is not None:
                        ts_file = pd.to_datetime(f.timestamp)
                    else:
                        ts_file = pd.to_datetime(f.timestamp, utc=True)
                    
                    if hasattr(pico_data['timestamp'], 'dt') and pico_data['timestamp'].dt.tz is not None:
                        pico_ts = pd.to_datetime(pico_data['timestamp'])
                    else:
                        pico_ts = pd.to_datetime(pico_data['timestamp'], utc=True)
                    
                    # Ensure both are in the same timezone for comparison
                    if ts_file.dt.tz != pico_ts.dt.tz:
                        if ts_file.dt.tz is not None and pico_ts.dt.tz is not None:
                            pico_ts = pico_ts.dt.tz_convert(ts_file.dt.tz)
                        elif ts_file.dt.tz is None and pico_ts.dt.tz is not None:
                            ts_file = ts_file.dt.tz_localize('UTC')
                            pico_ts = pico_ts.dt.tz_convert('UTC')
                        elif ts_file.dt.tz is not None and pico_ts.dt.tz is None:
                            pico_ts = pico_ts.dt.tz_localize('UTC').dt.tz_convert(ts_file.dt.tz)
                    # Determine file indices that fall within the pico timestamp range.
                    valid_mask = (ts_file >= pico_ts.min()) & (ts_file <= pico_ts.max())
                    ts_overlap = ts_file[valid_mask]
                    # Interpolate without extrapolating outside the pico range.
                    interp_pot = np.interp(
                    ts_overlap.astype('int64'),
                    pico_ts.astype('int64'),
                    pico_data['pot'],
                    left=np.nan,
                    right=np.nan
                    )
                    # Use the overlapping indices for all pico-related arrays.
                    faux['pico']['pot'] = interp_pot
                    faux['pico']['time'] = np.array(f.time)[valid_mask]
                    faux['pico']['timestamp'] = np.array(f.timestamp)[valid_mask]
                    faux['pico']['counter_pot'] = np.array(f.pot)[valid_mask] - interp_pot
                    if not valid_mask.any():
                        self.logger.debug('No matching pico data found for file %s', f.fname)
                    else:
                        self.logger.debug('Pico data associated with file %s', f.fname)
                        self.logger.debug('Column lengths: %d', len(interp_pot))
                except (ValueError, KeyError, TypeError) as e:
                    self.logger.error('Error processing pico data: %s', e)
                    self.logger.debug('Exception details:', exc_info=True)
            else:
                self.logger.warning('Pico data is missing the "timestamp" key or it is None.')
        else:
            self.logger.warning('No pico data available in auxiliary data.')

        # Process Furnace Data if Available
        if aux.get('furnace') is not None:
            
            furnace_data = aux['furnace']
            if 'timestamp' in furnace_data and furnace_data['timestamp'] is not None:
                try:
                    # Create a mask for data within the time span
                    furnace_mask = (furnace_data['timestamp'] >= tstart) & (furnace_data['timestamp'] <= tend)

                    # Apply the mask to each key in furnace data
                    for key, values in furnace_data.items():
                        if isinstance(values, np.ndarray):
                            faux['furnace'][key] = values[furnace_mask]
                        else:
                            self.logger.info('Expected numpy array for key "%s" in furnace data.', key)

                    # Calculate additional fields
                    faux['furnace']['time'] = (faux['furnace']['timestamp'] - tstart).astype('timedelta64[s]').astype(float)
                except Exception as e:
                    self.logger.error('Error processing furnace data: %s', e)
                    self.logger.debug('Exception details:', exc_info=True)
            else:
                self.logger.info('Furnace data is missing the "timestamp" key or it is None.')
        else:
            self.logger.info('No furnace data available in auxiliary data.')

        return faux

    def _process_collation_mapping(self, eclist: EcList, collation_mapping: Dict) -> EcList:
        """
        Process collation mapping to combine files into target classes before auxiliary data handling.
        
        Args:
            eclist: The EcList containing all loaded files
            collation_mapping: Dict mapping target classes to configurations for collating files.
                    Supports two formats:
                    1. Simple format: {target_class: {'id_numbers': [list], 'cyclic': bool, 'kwargs': dict}}
                    2. Enhanced format: {target_class: {'id_numbers': {id: kwargs_dict}, 'cyclic': bool}}
                    
        Returns:
            EcList: Updated EcList with collated files replacing original files
        """
        processed_eclist = eclist
        
        for target_class, config in collation_mapping.items():
            try:
                # Extract configuration
                id_numbers_config = config.get('id_numbers', [])
                cyclic = config.get('cyclic', False)
                default_kwargs = config.get('kwargs', {})
                
                if not id_numbers_config:
                    self.logger.warning('No id_numbers specified for target class %s, skipping', 
                                      target_class.__name__)
                    continue
                
                # Determine if we're using simple format (list) or enhanced format (dict)
                if isinstance(id_numbers_config, list):
                    # Simple format: use default kwargs for all id_numbers
                    id_number_mapping = {id_num: default_kwargs for id_num in id_numbers_config}
                elif isinstance(id_numbers_config, dict):
                    # Enhanced format: per-id_number kwargs
                    id_number_mapping = id_numbers_config
                else:
                    self.logger.error('Invalid id_numbers format for target class %s, must be list or dict', 
                                    target_class.__name__)
                    continue
                
                # Filter files for each id_number and collate them
                for id_number, specific_kwargs in id_number_mapping.items():
                    try:
                        # Filter files by id_number
                        filtered_files = processed_eclist.filter(id_number=id_number)
                        
                        if len(filtered_files) == 0:
                            self.logger.warning('No files found for id_number %s, skipping', id_number)
                            continue
                        
                        # Sort filtered files by starttime to ensure correct chronological order
                        try:
                            def get_sort_time(f):
                                """Get a sortable time value from file, handling different timestamp formats"""
                                if hasattr(f, 'starttime') and f.starttime is not None:
                                    return f.starttime
                                elif hasattr(f, 'timestamp') and f.timestamp is not None and len(f.timestamp) > 0:
                                    return f.timestamp[0]
                                else:
                                    return datetime.min  # Very early date as fallback
                            
                            filtered_files.sort(key=get_sort_time)
                            self.logger.debug('Sorted %d files for id_number %s by starttime before collation', 
                                            len(filtered_files), id_number)
                        except Exception as sort_error:
                            self.logger.warning('Could not sort files for id_number %s by starttime: %s. '
                                              'Proceeding with original order.', id_number, sort_error)
                        
                        # Log appropriate message based on number of files
                        if len(filtered_files) == 1:
                            self.logger.info('Converting single file for id_number %s to %s', 
                                           id_number, target_class.__name__)
                        else:
                            self.logger.info('Collating %d files for id_number %s into %s', 
                                           len(filtered_files), id_number, target_class.__name__)
                        
                        # Merge default kwargs with specific kwargs (specific takes precedence)
                        final_kwargs = {**default_kwargs, **specific_kwargs}
                        
                        # Collate and convert the filtered files (works for both single and multiple files)
                        converted_obj = filtered_files.collate_convert(
                            target_class=target_class, 
                            cyclic=cyclic, 
                            **final_kwargs
                        )
                        
                        # Replace the original files with the converted object
                        processed_eclist = processed_eclist.replace(converted_obj)
                        
                        if len(filtered_files) == 1:
                            self.logger.info('Successfully converted single file id_number %s to %s', 
                                           id_number, target_class.__name__)
                        else:
                            self.logger.info('Successfully collated id_number %s files into %s', 
                                           id_number, target_class.__name__)
                        
                    except Exception as e:
                        self.logger.error('Error collating files for id_number %s: %s', id_number, e)
                        continue
                        
            except Exception as e:
                self.logger.error('Error processing collation for target class %s: %s', 
                                target_class.__name__, e)
                continue
        
        return processed_eclist
