'''ectools_main.py'''
# Standard library imports
import logging
import os
import re
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd


# Relational imports
from .classes import EcList, ElectroChemistry
from .config import get_config
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
    def __init__(self, fname_parser=None, aux_importer=None, log_level="WARNING", **kwargs):
        '''
        fname_parser: optional function to parse information from the file name and path.
            Expected to return a dictionary, from which the key-value pairs are added to the 
            container object returned from load_file.
        log_level: logging level for the logger as a string (e.g., "DEBUG", "INFO", "ERROR").
        kwargs: key-val pairs added to this instance.
        '''
        self.fname_parser = fname_parser
        self.aux_importer = aux_importer
        self._setup_logging(log_level)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup_logging(self, log_level):
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

    def load_folder(self, fpath: str, data_folder_id=None, aux_folder_id=None, sort_by=None, **kwargs) -> EcList:
        '''
        Parse and load the contents of a folder and its subfolders.
        Ignores folders starting with '_' and only includes subfolders containing data_folder_id.
        
        Args:
            fpath: Root folder path to process
            data_folder_id: Override for data folder identifier (default from config)
            aux_folder_id: Override for aux folder identifier (default from config)
            sort_by: Attribute name to sort the EcList by (e.g., 'starttime', 'fname'). 
                    Default None (no sorting, files in order read)
            **kwargs: Additional arguments passed to EcList
        '''
        # Get folder identifiers from config or use overrides
        data_id = data_folder_id or get_config('data_folder_identifier') or 'data'
        aux_id = aux_folder_id or get_config('auxiliary_folder_identifier') or 'aux'
        
        self.logger.debug('Using folder identifiers - data: "%s", aux: "%s"', data_id, aux_id)
        
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
        eclist = EcList(fpath=fpath, **kwargs)
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
        
        if self.aux_importer:
            try:
                self.logger.info('Importing auxiliary data...')
                eclist.aux = self.aux_importer(fpath, aux_folder_id=aux_id)
            except RuntimeError as e:
                self.logger.warning('Error importing auxiliary data: %s', e)
                eclist.aux = None
            if eclist.aux is not None:
                try: # Attempt to associate auxiliary data with each file
                    for f in eclist:
                        f.aux = self._associate_auxiliary_data(eclist.aux, f)
                    self.logger.info('Successfully associated auxiliary data with %d files', len(eclist))
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    self.logger.exception('Error associating auxiliary data: %s', e)
        
        # Sort the EcList if sort_by parameter is provided
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
                    for key, id_tuple in container.column_patterns.items():
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
                for key, id_tuple in container.column_patterns.items():
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
            for class_ident in child.identifiers:
                if re.match(class_ident, ident):
                    return child
        return ElectroChemistry  # If no identifier is matched, return ElectroChemistry class

    def _associate_auxiliary_data(self, aux, f) -> Dict:
        '''Associate aux data with a file time span'''
        # Convert start and end timestamps
        tstart = np.datetime64(f.timestamp[0])
        tend = np.datetime64(f.timestamp[-1])

        # Initialize the faux dictionary with empty sub-dictionaries
        faux = {'pico': {}, 'furnace': {}}

        # Process Pico Data if Available
        if aux.get('pico'):
            pico_data = aux['pico']
            if 'timestamp' in pico_data and pico_data['timestamp'] is not None:
                try:
                    ts_file = pd.to_datetime(f.timestamp, utc=True).tz_convert(None)
                    pico_ts = pd.to_datetime(pico_data['timestamp'], utc=True).tz_convert(None)
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
