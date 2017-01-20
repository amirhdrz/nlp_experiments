"""
Utility Functions
"""

def print_red(text):
    print('\x1b[6;30;42m' + text + '\x1b[0m')

def print_green(text):
    print('\033[92m' + text + '\033[0m')

def print_y(text):
    print('\033[93m' + text + '\033[0m')
