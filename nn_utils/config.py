"""
config.py

Initial version by R. Mukai 16 January 2018

Configuration file to specify logging and other
central parameters.
"""

import logging

CONFIG_LOG_FILE = "D:\\Github\\logic_processor\\logs\\logfile.txt"

logging.basicConfig(filename=CONFIG_LOG_FILE, level=logging.INFO)

CONFIG_SYMBOL_LIST = [
    '(', ')', 'bookend', 'and', 'or', 'not', 'forall',
    'xor',
    '=', 'is', 'true', 'false', 'unknown', 'exists',
    'believes', 'knows', 'name', 'me', 'what', '.', '?',
    'if', 'then', 'help', "recite", None
]