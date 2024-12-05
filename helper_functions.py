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
        raise RuntimeError('Error reading json file') from e
    
    # Process 'oxide' samples if present in JSON data
    if 'oxide' in aux and 'salt_sampled' in aux:
        aux['oxide_measurents'] = aux['oxide']
        oxide = []
        salt_sampled = aux['salt_sampled']
        
        for measurements, timestamp in zip(aux['oxide'], salt_sampled):
            try:
                sample = OxideSample(measurements, timestamp)
                oxide.append(sample)
            except ValueError as e:
                pass
        
        aux['oxide'] = oxide
    else:
        aux['oxide_samples'] = []
    # Read pico file(s)
    aux['pico'] = {}
    try:
        pico_files = glob.glob(os.path.join(auxiliary_path, '**', '*pico*.csv'), recursive=True)
        if pico_files:
            pico_data = pd.concat([pd.read_csv(f'{file}', header=0) for file in pico_files])
            # Note: this will break if the data columns have different names (e.g. one is last 
            # and one is ave)
            # Convert the first column to datetime and rename it to 'Timestamp'
            aux['pico']['timestamp'] = pd.to_datetime(
                pico_data.iloc[:, 0], errors='coerce', unit='s').to_numpy()
            aux['pico']['pot'] = pico_data.iloc[:, 1].to_numpy()
            if ('mV' in pico_data.columns[1]):
                aux['pico']['pot'] = aux['pico']['pot'] / 1000
            elif ('V' in pico_data.columns[1]):
                pass
            else:
                raise ValueError("Cannot determine the unit of the potential data in pico CSV file. Expected 'mV' or 'V'.")
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
            cascade_data = pd.read_csv(os.path.join(cascade_path, csv_file), header=1)
            aux['furnace']['cascade_timestamp'] = pd.to_datetime(
                cascade_data['Date'] + ' ' + cascade_data['Time'], errors='coerce').to_numpy()
            aux['furnace']['cascade_celsius'] = cascade_data['Cascade_Controller_PV'].to_numpy()
            aux['furnace']['cascade_setpoint'] = cascade_data['Cascade_Controller_Working_SP'].to_numpy()
        else:
            aux['furnace']['cascade_celsius'] = None
    except Exception as e:
        raise RuntimeError(f'Error reading cascade controller csv file from {cascade_path}') from e
    
    # Read main controller csv file
    try:
        main_path = os.path.join(auxiliary_path, 'MainController')
        if os.path.exists(main_path):
            csv_file = [file for file in os.listdir(main_path) if file.endswith('.csv')][0]
            main_data = pd.read_csv(os.path.join(main_path, csv_file), header=1)
            aux['furnace']['main_timestamp'] = pd.to_datetime(
                main_data['Date'] + ' ' + main_data['Time'], errors='coerce').to_numpy()
            aux['furnace']['main_celsius'] = main_data['Main_Controller_PV'].to_numpy()
            aux['furnace']['main_setpoint'] = main_data['Main_Controller_Working_SP'].to_numpy()
        else:
            aux['furnace']['main_celsius'] = None
    except Exception as e:
        raise RuntimeError('Error reading main controller csv file') from e

    # Assert timestamps are equivalent and use single timestamp
    if aux['furnace']['cascade_celsius'] is not None and aux['furnace']['main_celsius'] is not None:
        if not np.array_equal(aux['furnace']['cascade_timestamp'],
        aux['furnace']['main_timestamp']):
            raise ValueError('Cascade and Main timestamps do not match')
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
def display_auxiliary_data(fl, oxide=True, furnace=True, pico=True):
    """todo"""
    if not fl.aux:
        print("No auxiliary data found")
        return
    text_out =  []
    temperature = fl.aux['temperature_celsius']
    co2_percentage = fl.aux['CO2_percent']
    run_id = fl.aux['run_id']
    text_out.append(f"Run {run_id}: {temperature}°C, {co2_percentage}% CO2")
    if oxide:
        oxide_samples = fl.aux['oxide']
        if oxide_samples:
            text_out.append("Oxide samples:")
            for i, sample in enumerate(oxide_samples):
                letter = chr(65 + i)
                text_out.append(f"{run_id}_{letter}: {sample.mean:.2f} ± {sample.stdev:.2f}, sampled {sample.timestamp}")
        else:
            text_out.append("No oxide samples recorder, or json file not properly formatted")
    for line in text_out:
        print(line)
    if furnace and pico:
        from bokeh.plotting import figure, show, output_notebook
        from bokeh.models import ColumnDataSource, HoverTool, DataRange1d, LinearAxis

        # Initialize Bokeh to display plots in the notebook
        output_notebook()
        # Assuming fl.aux['furnace'] and fl.aux['pico'] are existing pandas DataFrames
        # Create copies to avoid modifying the original data
        furnace_data = fl.aux['furnace'].copy()
        pico_data = fl.aux['pico'].copy()

        # Multiply 'pot' by 1000 to convert from V to mV and create a new column
        pico_data['pot_mV'] = pico_data['pot'] * 1000
        source_furnace = ColumnDataSource(data=furnace_data)
        source_pico = ColumnDataSource(data=pico_data)

        # Create a figure with a shared x-axis
        p = figure(title="Auxiliary Data",
                x_axis_label='Time',
                x_axis_type='datetime',
                width=800,
                height=400)

        cascade_line = p.line(x='timestamp', y='cascade_celsius', source=source_furnace, 
                            legend_label="Temperature", line_width=2, color='blue')

        # Configure left y-axis
        p.yaxis.axis_label = "Temperature"
        p.yaxis.axis_label_text_color = "blue"
        p.yaxis.major_label_text_color = "blue"

        # Add hover tool for Cascade_Controller_PV
        hover1 = HoverTool(renderers=[cascade_line],
                        tooltips=[
                            ("Timestamp", "@timestamp{%F %T}"),
                            ("Temperature", "@cascade_celsius")
                        ],
                        formatters={'@timestamp': 'datetime'})
        p.add_tools(hover1)

        # Define a new y-range for the right y-axis
        p.extra_y_ranges = {
            "pot_range": DataRange1d()
        }

        # Add the right y-axis associated with the 'pot_range'
        p.add_layout(LinearAxis(y_range_name="pot_range", axis_label="Cell Potential (V)", 
                                axis_label_text_color="red", major_label_text_color="red"), 'right')

        # Plot Cell Potential on the right y-axis
        pot_line = p.line(x='timestamp', y='pot_mV', source=source_pico, 
                        legend_label="Cell Potential", line_width=2, color='red', y_range_name="pot_range")

        # Add hover tool for Cell Potential
        hover2 = HoverTool(renderers=[pot_line],
                        tooltips=[
                            ("Timestamp", "@timestamp{%F %T}"),
                            ("Cell Potential (mV)", "@pot_mV")
                        ],
                        formatters={'@timestamp': 'datetime'})
        p.add_tools(hover2)

        # Configure y-axes labels and colors
        p.yaxis[0].axis_label = "Cascade Controller PV"
        p.yaxis[0].axis_label_text_color = "blue"
        p.yaxis[0].major_label_text_color = "blue"

        p.yaxis[1].axis_label = "Cell Potential (mV)"
        p.yaxis[1].axis_label_text_color = "red"
        p.yaxis[1].major_label_text_color = "red"

        # Show legend
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"  # Optional: allows interactive legend

        # Enhance layout
        p.xaxis.axis_label_text_font_size = "10pt"
        p.yaxis.axis_label_text_font_size = "10pt"
        p.title.text_font_size = "12pt"

        # Show the combined plot
        show(p)
    elif furnace or pico:
        warnings.warn("Not implemented")
