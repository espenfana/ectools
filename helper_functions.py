'''Electrochemistry related support module

Not integral part of ectools package

Custom scripts specific to BCS for interacting with ectools, extracting metadata and auxiliary data,
displaying and analyzing data.

'''

import re
import os
import glob
import json
from datetime import datetime
import logging
from typing import Dict, List
import warnings

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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

class PicoLogger():
    pass

class FurnaceLogger():
    pass

def mc_auxiliary_importer(fpath: str, aux_folder_id: str = None) -> Dict:
    """
    Load auxiliary data from JSON, furnace and pico files located in the specified file path.
    Parameters:
        fpath (str): The base file path where the auxiliary directory is located.
        aux_folder_id (str): Identifier for auxiliary folder (default: 'auxiliary')
    Returns:
        Dict: A dictionary containing the combined data from JSON and CSV files.
    Raises:
        AssertionError: If no auxiliary path with the identifier is found.
        RuntimeError: If there is an error reading the cascade or main controller CSV files.
    """
    from .config import get_config
    
    # Use provided identifier or get from config, fallback to 'auxiliary'
    folder_id = aux_folder_id or get_config('auxiliary_folder_identifier') or 'auxiliary'
    
    # Find all folders containing the identifier (partial match, case-insensitive)
    matching_folders = []
    try:
        for item in os.listdir(fpath):
            item_path = os.path.join(fpath, item)
            if os.path.isdir(item_path) and folder_id.lower() in item.lower():
                matching_folders.append((item, item_path))
    except (OSError, PermissionError):
        pass
    
    # Fallback to exact 'auxiliary' folder if no partial matches found
    if not matching_folders:
        auxiliary_path = os.path.join(fpath, 'auxiliary')
        if os.path.exists(auxiliary_path):
            matching_folders = [('auxiliary', auxiliary_path)]
        else:
            assert False, f"No auxiliary folders found matching '{folder_id}' or exact 'auxiliary' folder"
    
    logger.debug('Found %d auxiliary folders: %s', len(matching_folders), [folder[0] for folder in matching_folders])
    aux = {}

    # Read auxiliary json files from all matching folders
    try:
        for folder_name, folder_path in matching_folders:
            json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
            for file in json_files:
                with open(file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    aux.update(json_data)
                    logger.debug('Loaded JSON from: %s', file)
    except Exception as e:
        raise RuntimeError('Error reading json file') from e

    # Process 'oxide' samples if present in JSON data
    if 'oxide' in aux and 'salt_sampled' in aux:
        aux['oxide_measurements'] = aux['oxide']
        oxide = {}
        salt_sampled = aux['salt_sampled']

        for sample_id, measurements in aux['oxide'].items():
            try:
                sample = OxideSample(measurements, salt_sampled[sample_id])
                oxide[sample_id] = sample
            except ValueError as e:
                logger.warning("Error processing oxide sample %s: %s", sample_id, e)
            except KeyError as e:
                logger.warning(
                    "Error processing oxide sample. Probable mismatched sample and timestamp %s: %s",
                    sample_id, e
                )
        aux['oxide'] = oxide
    else:
        aux['oxide_samples'] = []
    
    # Read pico file(s) from all matching folders
    try:
        pico_files = []
        for folder_name, folder_path in matching_folders:
            pico_files.extend(glob.glob(os.path.join(folder_path, '**', '*pico*.csv'), recursive=True))
        
        if pico_files:
            pico_data = pd.concat([pd.read_csv(file, header=0) for file in pico_files])
            # Note: this will break if the data columns have different names (e.g. one is last 
            # and one is ave)
            # Convert the first column to datetime and rename it to 'Timestamp'
            aux['pico'] = {}
            aux['pico']['timestamp'] = pd.to_datetime(
                pico_data.iloc[:, 0], errors='coerce', unit='s').to_numpy()
            aux['pico']['pot'] = pico_data.iloc[:, 1].to_numpy()
            if ('mV' in pico_data.columns[1]):
                aux['pico']['pot'] = aux['pico']['pot'] / 1000
            elif ('V' in pico_data.columns[1]):
                pass
            else:
                raise ValueError(f"Cannot determine the unit of the potential data in pico CSV file. Expected 'mV' or 'V'. Found '{pico_data.columns[1]}' instead.")
        else:
            aux['pico'] = None
    except Exception as e:
        raise RuntimeError('Error reading pico csv file') from e

    # Read cascade controller csv files from all matching folders
    aux['furnace'] = {}
    cascade_data_list = []
    try:
        for folder_name, folder_path in matching_folders:
            cascade_path = os.path.join(folder_path, 'CascadeController')
            if os.path.exists(cascade_path):
                csv_files = [file for file in os.listdir(cascade_path) if file.endswith('.csv')]
                for csv_file in csv_files:
                    cascade_data = pd.read_csv(os.path.join(cascade_path, csv_file), header=1)
                    cascade_data['timestamp'] = pd.to_datetime(
                        cascade_data['Date'] + ' ' + cascade_data['Time'], errors='coerce')
                    cascade_data_list.append(cascade_data)
                    logger.debug('Loaded cascade data from: %s', os.path.join(cascade_path, csv_file))
        
        if cascade_data_list:
            # Combine all cascade data and sort by timestamp
            combined_cascade = pd.concat(cascade_data_list, ignore_index=True).sort_values('timestamp')
            aux['furnace']['cascade_timestamp'] = combined_cascade['timestamp'].to_numpy()
            aux['furnace']['cascade_celsius'] = combined_cascade['Cascade_Controller_PV'].to_numpy()
            aux['furnace']['cascade_setpoint'] = combined_cascade['Cascade_Controller_Working_SP'].to_numpy()
        else:
            aux['furnace']['cascade_celsius'] = None
    except Exception as e:
        raise RuntimeError(f'Error reading cascade controller csv files') from e
    
    # Read main controller csv files from all matching folders
    main_data_list = []
    try:
        for folder_name, folder_path in matching_folders:
            main_path = os.path.join(folder_path, 'MainController')
            if os.path.exists(main_path):
                csv_files = [file for file in os.listdir(main_path) if file.endswith('.csv')]
                for csv_file in csv_files:
                    main_data = pd.read_csv(os.path.join(main_path, csv_file), header=1)
                    main_data['timestamp'] = pd.to_datetime(
                        main_data['Date'] + ' ' + main_data['Time'], errors='coerce')
                    main_data_list.append(main_data)
                    logger.debug('Loaded main data from: %s', os.path.join(main_path, csv_file))
        
        if main_data_list:
            # Combine all main data and sort by timestamp
            combined_main = pd.concat(main_data_list, ignore_index=True).sort_values('timestamp')
            aux['furnace']['main_timestamp'] = combined_main['timestamp'].to_numpy()
            aux['furnace']['main_celsius'] = combined_main['Main_Controller_PV'].to_numpy()
            aux['furnace']['main_setpoint'] = combined_main['Main_Controller_Working_SP'].to_numpy()
        else:
            aux['furnace']['main_celsius'] = None
    except Exception as e:
        raise RuntimeError('Error reading main controller csv files') from e

    # Compare lengths first, cut off the longest array until lengths match
    if aux['furnace']['cascade_celsius'] is not None and aux['furnace']['main_celsius'] is not None:
        while len(aux['furnace']['cascade_timestamp']) != len(aux['furnace']['main_timestamp']):
            if len(aux['furnace']['cascade_timestamp']) > len(aux['furnace']['main_timestamp']):
                aux['furnace']['cascade_timestamp'] = aux['furnace']['cascade_timestamp'][:-1]
                aux['furnace']['cascade_celsius']   = aux['furnace']['cascade_celsius'][:-1]
            else:
                aux['furnace']['main_timestamp'] = aux['furnace']['main_timestamp'][:-1]
                aux['furnace']['main_celsius']   = aux['furnace']['main_celsius'][:-1]
        # If timestamps differ when lengths are equal, discard furnace data
        if not np.array_equal(aux['furnace']['cascade_timestamp'], aux['furnace']['main_timestamp']):
            logger.warning("Furnace data was read, but failed parsing.")
            aux['furnace'] = None
        else:
            aux['furnace']['timestamp'] = aux['furnace']['cascade_timestamp']
            del aux['furnace']['cascade_timestamp']
            del aux['furnace']['main_timestamp']

    return aux


class OxideSample:
    """
    Represents an oxide sample with multiple measurements and a corresponding timestamp.
    
    Attributes:
        measurements (List[float]): A list of individual measurement values.
        timestamp (datetime): The timestamp when the sample was taken.
    """
    
    def __init__(self, measurements: List[float], timestamp: str):
        """
        Initializes the OxideSample instance.
        
        Args:
            measurements (List[float]): A list of measurement values.
            timestamp (str): The timestamp as a string. It will be converted to a datetime object.
        
        Raises:
            ValueError: If measurements list is empty or contains non-numeric values.
            ValueError: If timestamp string is not in a recognizable datetime format.
        """
        if not measurements:
            raise ValueError("Measurements list cannot be empty.")
        if not all(isinstance(m, (int, float)) for m in measurements):
            raise ValueError("All measurements must be numeric values.")
        
        try:
            self.timestamp = datetime.fromisoformat(timestamp)
        except ValueError as ve:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from ve
        
        self.measurements = measurements
    
    @property
    def mean(self) -> float:
        """Calculates and returns the mean of the measurements."""
        return np.mean(self.measurements)
    
    @property
    def stdev(self) -> float:
        """Calculates and returns the standard deviation of the measurements."""
        return np.std(self.measurements, ddof=1)  # Sample standard deviation
    
    def __str__(self) -> str:
        """
        Returns a formatted string representation of the OxideSample.
        
        Format:
            "Oxide samples: mean ± stdev, sampled 'timestamp', ([measurements])"
        """
        return (f"Oxide sample: {self.mean:.3f} ± {self.stdev:.3f}, "
                f"sampled '{self.timestamp}', {self.measurements}")

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the OxideSample."""
        return (f"OxideSample(measurements={self.measurements}, "
                f"timestamp='{self.timestamp.isoformat()}')")

if __name__ == '__main__':
    # Test the filename parser
    TEST_FNAME = '241021_21_MCL23_WE2_CV2_03CO2_750C.DTA'
    print(mc_filename_parser(None, TEST_FNAME))

# ---   Display auxiliary data   --- #
def display_auxiliary_data(fl, oxide=True, furnace=True, pico=True, heating_rate=False):
    """Displays auxiliary data and optional plots for furnace, pico, and heating rate."""
    text_out =  []
    temperature = fl.aux['temperature_celsius']
    co2_percentage = fl.aux['CO2_percent']
    run_id = fl.aux['run_id']
    text_out.append(f"Run {run_id}: {temperature}°C, {co2_percentage}% CO2")
    if oxide:
        oxide_samples = fl.aux['oxide']
        if oxide_samples:
            text_out.append("Oxide samples:")
            for suffix, sample in oxide_samples.items():
                text_out.append(f"{run_id}_{suffix}: {sample.mean:.2f} ± {sample.stdev:.2f}, sampled {sample.timestamp}")
        else:
            text_out.append("No oxide samples recorded, or json file not properly formatted")
    for line in text_out:
        print(line)
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d
    output_notebook()

    # Separate Furnace Plot with Dual y-Axes (Temperature and Heating Rate)
    if furnace and fl.aux.get('furnace') is not None:
        furnace_data = fl.aux['furnace'].copy()
        
        # Compute heating rate for cascade temperature
        ts = np.array(furnace_data['timestamp'])
        cascade = np.array(furnace_data['cascade_celsius'])
        ts_sec = ts.astype("datetime64[s]").astype("int64")
        dt = np.diff(ts_sec) / 60.0  # minutes
        d_t = np.diff(cascade)
        heating_rate = np.concatenate(([np.nan], d_t / dt))
        furnace_data["heating_rate"] = heating_rate

        # Compute heating rate for main temperature if available
        if "main_celsius" in furnace_data:
            main = np.array(furnace_data["main_celsius"])
            dT_main = np.diff(main)
            heating_rate_main = np.concatenate(([np.nan], dT_main / dt))
            furnace_data["heating_rate_main"] = heating_rate_main

        # Create data source with all data for comprehensive tooltips
        temp_data = {
            'timestamp': furnace_data['timestamp'],
            'cascade_celsius': furnace_data['cascade_celsius'],
            'heating_rate': furnace_data['heating_rate']
        }
        if "main_celsius" in furnace_data:
            temp_data['main_celsius'] = furnace_data['main_celsius']
            temp_data['heating_rate_main'] = furnace_data['heating_rate_main']

        source_temp = ColumnDataSource(data=temp_data)

        p_furnace = figure(title="Furnace Data",
                           x_axis_label='Time',
                           x_axis_type='datetime',
                           width=800,
                           height=400)
        
        # Plot temperature data on left y-axis
        temp_lines = []
        temp_lines.append(p_furnace.line(x='timestamp', y='cascade_celsius', source=source_temp,
                                        legend_label='Thermocouple', line_width=2, color="blue"))
        
        if "main_celsius" in furnace_data:
            temp_lines.append(p_furnace.line(x='timestamp', y='main_celsius', source=source_temp,
                                            legend_label='Heating Element', line_width=2, color='orange'))

        # Add extra y-axis for heating rate
        p_furnace.extra_y_ranges = {"heating_rate": Range1d(start=-20, end=40)}
        p_furnace.add_layout(LinearAxis(y_range_name="heating_rate", axis_label="Heating Rate (°C/min)"), 'right')
        p_furnace.y_range = Range1d(start=-400, end=800)
        
        # Plot heating rate data on right y-axis using same source
        p_furnace.line(x='timestamp', y='heating_rate', source=source_temp,
                      legend_label="Heating Rate (Cascade)", line_width=2, color="green",
                      y_range_name="heating_rate")
        
        if "main_celsius" in furnace_data:
            p_furnace.line(x='timestamp', y='heating_rate_main', source=source_temp,
                          legend_label="Heating Rate (Main)", line_width=2, color="red",
                          y_range_name="heating_rate")

        # Create comprehensive tooltips with all data
        temp_tooltips = [
            ("Time", "@timestamp{%F %T}"),
            ("Cascade (°C)", "@cascade_celsius{0.0}"),
            ("Heating Rate Cascade (°C/min)", "@heating_rate{0.00}")
        ]
        if "main_celsius" in furnace_data:
            temp_tooltips.extend([
                ("Main (°C)", "@main_celsius{0.0}"),
                ("Heating Rate Main (°C/min)", "@heating_rate_main{0.00}")
            ])

        # Add hover tool only to temperature lines
        hover_temp = HoverTool(
            renderers=temp_lines,
            tooltips=temp_tooltips,
            formatters={'@timestamp': 'datetime'},
            mode='vline'
        )
        p_furnace.add_tools(hover_temp)

        show(p_furnace)
    else:
        print("No furnace data found")

    # Separate Pico Plot; if furnace data exists, align x_range to its timestamp axis.
    if pico and fl.aux.get('pico') is not None:
        pico_data = fl.aux['pico'].copy()
        pico_data['pot_mV'] = pico_data['pot']
        p_pico = figure(title="Pico Data",
                        x_axis_label='Time',
                        x_axis_type='datetime',
                        width=800,
                        height=400)
        source_pico = ColumnDataSource(data=pico_data)
        p_pico.line(x='timestamp', y='pot_mV', source=source_pico,
                    legend_label='Pico', line_width=2, color='green', name='pico_line')
        hover_pico = HoverTool(renderers=[p_pico.select_one({'name': 'pico_line'})],
                               tooltips=[
                                   ("Time", "@timestamp{%F %T}"),
                                   ("Pico (mV)", "@pot_mV")
                               ],
                               formatters={'@timestamp': 'datetime'},
                               mode='vline')
        p_pico.add_tools(hover_pico)
        p_pico.yaxis.axis_label = "Cell potential (V)"
        # Align pico x-axis to furnace x-axis if available
        if furnace and fl.aux.get('furnace') is not None:
            p_pico.x_range = p_furnace.x_range
        show(p_pico)
    else:
        print("No pico data found")
