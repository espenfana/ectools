'''ectools_main.py'''
import logging
import os
import re
import numpy as np
import pandas as pd

# Relational imports
from .classes import EcList, ElectroChemistry 
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
    def __init__(self, fname_parser=None, log_level="ERROR", **kwargs):
        '''
        fname_parser: optional function to parse information from the file name and path.
            Expected to return a dictionary, from which the key-value pairs are added to the 
            container object returned from load_file.
        log_level: logging level for the logger as a string (e.g., "DEBUG", "INFO", "ERROR").
        kwargs: key-val pairs added to this instance.
        '''
        self.fname_parser = fname_parser

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set logger to capture all levels, handlers will filter

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
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def load_folder(self, fpath: str, **kwargs) -> EcList:
        '''
        Parse and load the contents of a folder (not subfolders)
        '''
        flist = os.listdir(fpath)
        eclist = EcList(fpath=fpath, **kwargs)
        ignored = 0
        for i, fname in enumerate(flist):
            try:
                f = self.load_file(fpath, fname)
                if f:
                    eclist.append(f)
                else:
                    ignored += 1
            except Exception as error:  # pylint: disable=broad-except
                self.logger.error('Error processing file %s: %s', fname, error)
            finally:
                print(f'\rProcessing {i} of {len(flist)}' + '.' * (i % 7 + 1), end='\r')
        print(f'\nProcessed {len(flist)} entries, parsed {len(eclist)}, ignored {ignored}')
        #self.logger.info('Processed %d files, parsed %d', len(flist), len(eclist))
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
                self.logger.info('Successful import: %s', fname)
                return container
        except (IOError, OSError):
            self.logger.info('File cannot be opened, skipping: %s', fname)
            return None
        except Exception as error:  # pylint: disable=broad-except
            self.logger.error('Error loading file %s: %s', fname, error)
            raise error

    def _parse_file_gamry(self, fname, fpath):
        '''Parse a Gamry formatted ascii file (such as .-DAT). 
        Returns a custom electrochemistry container object '''
        meta_list = []
        try:
            with open(os.path.join(fpath, fname), encoding="utf-8") as f:
                # Open the file to read the first lines
                technique = None
                while line := f.readline():
                    if line == "": # at EOF, readline() will return an empty string
                        raise RuntimeError('No TABLE detected')
                    line = line.rstrip().split('\t')
                    if 'TABLE' in line:
                        if 'OCVCURVE' in line: # TODO add saving of OCP curve
                            num_lines = int(line[-1])
                            f.readlines(num_lines)
                        else:
                            break
                    meta_list.append(line)
                    if 'TAG' in line:
                        technique = line[1]

                #next two lines should be header and units
                header_row = f.readline().split('\t')
                units_row = f.readline().split('\t')
                # Grabbing the technique from the metadata, we can use the appropriate container object
                container_class = self._get_class(technique)
                container = container_class(fname, fpath, meta_list)

                coln = {} # identifier to column number dictionary
                units = {} # identifier to column unit dictionary
                for i, column_header in enumerate(header_row):
                    for key, id_tuple in container.get_columns.items():
                        for id_rgx in id_tuple:
                            if re.match(id_rgx, column_header):
                                coln[key] = i
                                units[key] = units_row[i]

                data_block = []
                if technique == 'CV':
                    # gamry CV's require custom handing due to "CURVE"s being separate tables
                    cycle_list = []
                    ncycle = 1
                    while line := f.readline():
                        if line[:5] == 'CURVE':
                            ncycle +=1
                            assert f.readline().split('\t') == header_row
                            assert f.readline().split('\t') == units_row
                            continue
                        data_block.append(line.split('\t'))
                        cycle_list.append(ncycle)
                else: # assuming other types have a single data table
                    data_block = [line.split('\t') for line in f.readlines()]

                for key, j in coln.items():
                    container[key] = np.array([line[j] for line in data_block], dtype='float')
                if technique == 'CV':
                    container['cycle'] = np.array(cycle_list, dtype='int')
            container.units = units
            container.parse_meta_gamry()

            #container['realtime'] = np.array(container['time'], dtype=datetime) + container.starttime
            if any(container.curr):
                container.curr_dens = np.divide(container.curr, container.area)
            return container
        except Exception as error:  # pylint: disable=broad-except
            error_message = f'-F- {fpath}{fname}\necImporter.parse_file_gamry error\n{error}'
            self.logger.error(error_message)
            raise error

    def _parse_file_mpt(self, fname, fpath):
        '''Parse an EC-lab ascii file. 
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
                for key, id_tuple in container.get_columns.items():
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

    def _get_class(self, ident: str):
        '''Tries to match the identifier with the container classes available.'''
        if ident is None:
            return ElectroChemistry
        for child in ElectroChemistry.__subclasses__():
            for class_ident in child.identifiers:
                if re.match(class_ident, ident):
                    return child
        return ElectroChemistry  # If no identifier is matched, return ElectroChemistry class
