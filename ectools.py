'''ectools.py'''
import numpy as np
import pandas as pd
import re
import os

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
            except:
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
            with open(fpath + fname) as f:
                row_1 = f.readline().strip()
                if re.match('EC-Lab ASCII FILE', row_1): # File identified as EC-lab
                    container = parse_file_mpt(fname, fpath)
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


def parse_file_mpt(fname, fpath):
    '''Parse an EC-lab ascii file. Returns a container object'''
    try:
        meta_list = []
        with open(fpath + fname) as f: # Open the file to read the first lines
            meta_list.append(f.readline().strip()) # EC-Lab
            meta_list.append(f.readline().strip()) # Contains the number of lines in the metadata block
            head_row = int(re.findall(r'\d\d', meta_list[1])[0]) -1 # Header row
            for _ in range(2, head_row+1):
                meta_list.append(f.readline().strip()) # Read the rest of the metadata block
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
