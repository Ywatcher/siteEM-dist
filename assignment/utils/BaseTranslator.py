from collections import namedtuple
from typing import Union, List
from enum import Enum

class BaseEnum(Enum):
    """
    An enumeration of DNA bases to integer indices
    """
    A = 0
    C = 1
    G = 2
    T = 3

class BaseTranslator:
    """
    A class to translate between string and integer representations of DNA
    bases
    :Example:
    >>> bt = BaseTranslator()
    >>> bt.translate("A")
    0
    >>> bt.translate(0)
    'A'
    """
    def __init__(self):
        self.base_to_index = BaseEnum
        self.index_to_base = {e.value: e.name for e in BaseEnum}
 
    def translate_int_to_char(self, num_str: List[int]) -> str:
        """
        Convert a list of numbers to a string of base characters.

        :param num_str: A string of numbers (e.g., '012')
        :type num_str: str
        
        :return: A string of base characters (e.g., 'ACG')
        :rtype: str
        
        :raises KeyError: If num_str contains invalid base indices
        """
        try:
            return "".join(self.index_to_base[num_char] for num_char in num_str)
        except KeyError as exc:
            exc.args = (
                f"{num_str} contains an invalid base index. Valid base indices are 0, 1, 2, 3.",
            )
            raise exc
    
    def translate_char_to_int(self, char_str: str) -> List[int]:
        """
        Convert a string of base characters to a list of integers.

        :param char_str: A string of base characters (e.g., 'ACG')
        :type char_str: str

        :return: A string of numbers (e.g., '012')
        :rtype: str
        
        :raises KeyError: If char_str contains invalid base characters
        """
        try:
            return [self.base_to_index[char].value 
                    for char in char_str.upper()]

        except KeyError as exc:
            exc.args = (
                f"{char_str} contains an invalid base. Valid bases are A, C, G, T.",
            )
            raise exc