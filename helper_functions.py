'''Electrochemistry related support module'''

import re

def filename_parser(_, fname: str) -> dict:
    """Parse the filename and return attribure dictionary. Expects a format like:
    '240926_15E_MCL19_cswWE1_SCAN-HOLD_LSV_STRIP_CO2_750C.DTA'
        Extracts:
            id_number: int
            id_letter: (optional) str
            we_number: int
            temperature: int
            mcl_number: int"""
    out = {}
    try:
        value_id = re.search(r'\d{6}_([0-9]+)([A-Za-z]*?)_', fname)
        value_id_full = re.search(r'(\d{6}_[0-9]+[A-Za-z]*?)_', fname)
        value_we_number = re.search(r'WE(\d+)', fname)
        value_temperature = re.search(r'_(\d+)C\.DTA', fname)
        mcl_number = re.search(r'_MCL(\d+)', fname)

        if value_id:
            out['id_number'] = int(value_id.group(1))
            out['id_letter'] = str(value_id.group(2)).lower() if value_id.group(2) else ''
            out['fid'] = str(out['id_number']) + out['id_letter']
        out['id_full'] = str(value_id_full.group(1))
        out['we_number'] = int(value_we_number.group(1)) if value_we_number else None
        out['temperature'] = int(value_temperature.group(1)) if value_temperature else None
        out['mcl_number'] = int(mcl_number.group(1)) if mcl_number else None
    except Exception:  # pylint: disable=broad-except
        pass
    return out


if __name__ == '__main__':
    # Test the filename parser
    test_fname = '241021_21_MCL23_WE2_CV2_1CO2_750C.DTA'
    print(filename_parser(None, test_fname))