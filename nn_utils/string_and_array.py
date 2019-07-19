"""
string_and_array.py

R. Mukai 11 September 2018

Goal of this module is to provide a utility to switch between a sentence string
and an array of symbols.
"""

def array_to_string(input_array):
    """
    array_to_string: Change an input array into a symbol string.
    :param input_array:
    :return: Return string
    """

    return_string = ""

    for symbol in input_array:

        if symbol is not None:

            return_string = return_string + " " + str(symbol)

    return return_string

# End function array_to_string