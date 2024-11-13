'''Electrochemistry related support module'''

import re
import os
import glob
import json
from typing import Dict, Tuple

import pandas as pd
import numpy as np

def mc_filename_parser(_, fname: str) -> dict:
    """Parse the filename and return attribute dictionary. Expects a format like:
    '240926_15E_MCL19_cswWE1_SCAN-HOLD_LSV_STRIP_CO2_750C.DTA'
        Extracts:
            id_number: int
            id_letter: (optional) str
            id: str
            id_full: str
            we_number: int
            temperature: int
            mcl_number: int
            co2_number: float"""
    out = {}
    value_id = re.search(r'\d{6}_([0-9]+)([A-Za-z]*?)_', fname)
    value_id_full = re.search(r'(\d{6}_[0-9]+[A-Za-z]*?)_', fname)
    value_we_number = re.search(r'WE(\d+)', fname)
    value_temperature = re.search(r'_(\d+)C\.DTA', fname)
    mcl_number = re.search(r'_MCL(\d+)', fname)
    co2_number = re.search(r'_(\d+)CO2', fname)

    if value_id:
        out['id_number'] = int(value_id.group(1))
        out['id_letter'] = str(value_id.group(2)).lower() if value_id.group(2) else ''
        out['id'] = str(out['id_number']) + out['id_letter']
    out['id_full'] = str(value_id_full.group(1)) if value_id_full else None
    out['we_number'] = int(value_we_number.group(1)) if value_we_number else None
    out['temperature'] = int(value_temperature.group(1)) if value_temperature else None
    out['mcl_number'] = int(mcl_number.group(1)) if mcl_number else None

    if co2_number:
        # because 03 is 0.3 and 1 is 1.0
        co2_str = co2_number.group(1)
        if co2_str.startswith('0') and len(co2_str) > 1:
            out['co2_number'] = float(f"0.{co2_str[1:]}")
        else:
            out['co2_number'] = float(co2_str)
    else:
        out['co2_number'] = None

    return out

def mc_auxiliary_importer(fpath: str) -> Dict:
    """
    Load auxiliary data from JSON, furnace and pico files located in the specified file path.
    Parameters:
        fpath (str): The base file path where the 'auxiliary' directory is located.
    Returns:
        Dict: A dictionary containing the combined data from JSON and CSV files.
    Raises:
        AssertionError: If the 'auxiliary' path does not exist.
        RuntimeError: If there is an error reading the cascade or main controller CSV files.
    """

    auxiliary_path = os.path.join(fpath, 'auxiliary')

    assert os.path.exists(auxiliary_path), f"Path {auxiliary_path} does not exist."
    aux = {}

    # Read auxiliary json files
    try:
        json_files = glob.glob(os.path.join(auxiliary_path, '**', '*.json'), recursive=True)
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                aux.update(json_data)
    except Exception as e:
        raise RuntimeError('Error reading json file')
    
    # Read pico file(s)
    aux['pico'] = {}
    try:
        pico_files = glob.glob(os.path.join(auxiliary_path, '**', '*pico*.csv'), recursive=True)
        if pico_files:
            pico_data = pd.concat([pd.read_csv(f'{file}', header=0) for file in pico_files])
            # Convert the first column to datetime and rename it to 'Timestamp'
            aux['pico']['timestamp'] = pd.to_datetime(
                pico_data.iloc[:, 0], errors='coerce', unit='s').to_numpy()
            aux['pico']['pot'] = pico_data.iloc[:, 1].to_numpy()
            if ('mV' in pico_data.columns[1]):
                aux['pico']['pot'] = aux['pico']['pot'] / 1000
            elif ('V' in pico_data.columns[1]):
                aux['pico']['pot'] = aux['pico']['pot']
            else:
                raise ValueError('Cannot determine the unit of the potential data')
        else:
            aux['pico'] = None
    except Exception as e:
        raise RuntimeError('Error reading pico csv file') from e

    # Read cascade controller csv file
    aux['furnace'] = {}
    try:
        cascade_path = os.path.join(auxiliary_path, 'CascadeController')
        if os.path.exists(cascade_path):
            csv_file = [file for file in os.listdir(cascade_path) if file.endswith('.csv')][0]
            cascade_data = pd.read_csv(f'auxiliary/CascadeController/{csv_file}', header=1)
            aux['furnace']['cascade_timestamp'] = pd.to_datetime(
                cascade_data['Date'] + ' ' + cascade_data['Time'], errors='coerce').to_numpy()
            aux['furnace']['cascade_celcius'] = cascade_data['Cascade_Controller_PV'].to_numpy()
            aux['furnace']['cascade_setpoint'] = cascade_data['Cascade_Controller_Working_SP'].to_numpy()
        else:
            aux['furnace']['cascade_celcius'] = None
    except Exception as e:
        raise RuntimeError('Error reading cascade controller csv file') from e
    
    # Read main controller csv file
    try:
        main_path = os.path.join(auxiliary_path, 'MainController')
        if os.path.exists(main_path):
            csv_file = [file for file in os.listdir(main_path) if file.endswith('.csv')][0]
            main_data = pd.read_csv(f'auxiliary/MainController/{csv_file}', header=1)
            aux['furnace']['main_timestamp'] = pd.to_datetime(
                main_data['Date'] + ' ' + main_data['Time'], errors='coerce').to_numpy()
            aux['furnace']['main_celcius'] = main_data['Main_Controller_PV'].to_numpy()
            aux['furnace']['main_setpoint'] = main_data['Main_Controller_Working_SP'].to_numpy()
        else:
            aux['furnace']['main_celcius'] = None
    except Exception as e:
        raise RuntimeError('Error reading main controller csv file') from e

    # Assert timestamps are equivalent and use single timestamp
    if aux['furnace']['cascade_celcius'] is not None and aux['furnace']['main_celcius'] is not None:
        if not np.array_equal(aux['furnace']['cascade_timestamp'],
        aux['furnace']['main_timestamp']):
            raise ValueError('Cascade and Main timestamps do not match')
        aux['furnace']['timestamp'] = aux['furnace']['cascade_timestamp']
        del aux['furnace']['cascade_timestamp']
        del aux['furnace']['main_timestamp']

    return aux

if __name__ == '__main__':
    # Test the filename parser
    TEST_FNAME = '241021_21_MCL23_WE2_CV2_03CO2_750C.DTA'
    print(mc_filename_parser(None, TEST_FNAME))
