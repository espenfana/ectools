'''ectools.py'''
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

# Relational imports
from . import classes # classes is a collection of container object classes meant for different electrochemical methods 

class ecImporter():
    '''Helper object for importing electrochemistry files'''
    def __init__(self, fname_parser: object=None , **kwargs):
        '''
        fname_parser: optional function to parse information from the file name and path.
            Expected to return a dictionary, from which the key-value pairs are added to the 
            container object returned from load_file.
        kwargs: key-val pairs added to this instance.
        '''
        self.fname_parser = fname_parser
        self.log = []
        for key, val in kwargs:
            setattr(self, key, val)
    

    def load_folder(self, fpath : str, **kwargs) -> classes.ecList:
        '''
        Parse and load the contents of a folder (not subfolders)
        '''
        flist = os.listdir(fpath)
        eclist = classes.ecList(fpath=fpath, **kwargs)
        for i, fname in enumerate(flist):
            try:
                f = self.load_file(fpath, fname)
                if f:
                    eclist.append(f)
            except Exception as E:
                print(E) # TODO testing
                pass
            finally:
                print(f'\rProcessing {i} of {len(flist)}' + '.'*(i%7+1), end='\r')
        print(f'Processed {len(flist)} files, parsed {len(eclist)}')

            
        return eclist
    
    def load_file(self, fpath, fname):
        '''Load and parse an electrochemistry file. Opens the file to look for an identifier, then passes 
        it along to the correct file parser.
        '''
        try:
            with open(os.path.join(fpath, fname)) as f:
                row_1 = f.readline().strip()
                if re.match('EC-Lab ASCII FILE', row_1): # File identified as EC-lab
                    container = parse_file_mpt(fname, fpath)
                elif re.match('EXPLAIN', row_1):
                    container = parse_file_gamry(fname, fpath)
                else:
                    self.log.append('-F- ' +  fpath + fname)
                    self.log.append('Not a recognized format')
                    return
                
                if self.fname_parser:
                    fname_dict = self.fname_parser(fpath, fname)
                    for key, val in fname_dict.items():
                        container[key] = val
                self.log.append('-S- ' + fpath + fname)
                return container
        except Exception as E:
            self.log.append('-F- ' +  fpath + fname)
            self.log.append('ecImporter.load_file error')
            self.log.append(E)
            raise E
    
       
def get_class(ident : str):
    '''Tries to match the identifier with the container classes available.'''
    for child in classes.ElectroChemistry.__subclasses__():
        for class_ident in child.identifiers:
            if re.match(class_ident, ident):
                return child
    return classes.ElectroChemistry # If no child indentifier is matched, return parent ElectroChemistry class

def parse_file_gamry(fname, fpath):
    '''Parse a Gamry formatted ascii file (such as .-DAT). Returns a custom electrochemistry container object '''
    meta_list = []
    try:
        with open(os.path.join(fpath, fname)) as f: # Open the file to read the first lines
            while line:=f.readline():

                if line == "": # at EOF, readline() will return an empty string
                    raise Exception('No TABLE detected')
                    
                line = line.rstrip().split('\t')
                if 'TABLE' in line: break 
                meta_list.append(line)

                if 'TAG' in line:
                    technique = line[1]
            
            #next two lines should be header and units
            header_row = f.readline().split('\t')
            units_row = f.readline().split('\t')
            # Grabbing the electrochemistry technique from the metadata, we can use the appropriate container object
            container_class = get_class(technique)
            #print(container_class) # TODO for testing
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
            container.units['curr_dens'] = f'{container.units["curr"]}/{container.units["area"]}'
        return container         
    except Exception as E:
        print('ectools.parse_file_gamry error:')
        raise E

def parse_file_mpt(fname, fpath):
    '''Parse an EC-lab ascii file. Returns a custom electrochemistry container object container object'''
    try:
        meta_list = []
        with open(os.path.join(fpath, fname)) as f: # Open the file to read the first lines
            meta_list.append(f.readline().strip()) # EC-Lab
            meta_list.append(f.readline().strip()) # Contains the number of lines in the metadata block
            head_row = int(re.findall(r'\d\d', meta_list[1])[0]) -1 # Header row
            for _ in range(2, head_row+1):
                meta_list.append(f.readline().strip()) # Read the rest of the metadata block

        # Grabbing the electrochemistry technique from the metadata, we can use the appropriate container object
        technique = meta_list[3] # The fourth line of the block should contain the EC-Lab technique used
        container_class = get_class(technique) # Match the technique to the container classes
        container = container_class(fname, fpath, meta_list) # Initialize the container
                
        headers = meta_list[-1].split('\t') # Isolate header row and split the string

        coln = {}
        units = {}
        # Match the columns the container class expects with the columns in the header. Need to maintain order as usecols does not!
        for i, colh in enumerate(headers):
            for key, id_tuple in container.get_columns.items():
                for id_rgx in id_tuple:
                    m = re.match(id_rgx, colh)
                    if m:
                        coln[key] = i
                        if m.groups():
                            units[key] = m.groups()[-1]
                        continue

        # Using pandas.read_csv because it is faster than any other csv data block importer methods i've tried AND it interprets data types
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
        del(df)
        container.units = units
        container.parse_meta_mpt()

        return container
    except Exception as E:
        print('ectools.parse_mpt error:')
        raise E
    return 
