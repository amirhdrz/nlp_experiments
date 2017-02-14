""" Methods in this file read data files and present them in an appropariate data structure.
"""

import json

# TODO: need a more robust implementation

def building_codes():
    """ Returns a dictionary keyed with building codes.
        Each key returns a tuple (building name, address)
    :return: dict of building codes
    """
    with open('./data_files/building_codes.json', 'r') as f:
        data = json.load(f)

    codes = {}
    for item in data:
        codes[item['Code']] = (item['Building'], item['Address'])

    return codes
