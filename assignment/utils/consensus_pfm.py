from .BaseTranslator import BaseTranslator
from typing import List


def consensus_pfm(pfm: List[List[float]]) -> str:
    """Given a PFM, return the most likely base (character representation) at
        each position. Where there are ties, the first base in the alphabet
        will be returned.

    :param pfm: A PFM represented as a list of lists of integers where each
        inner list represents the probability of observing a given base at
        that position in the motif.
    :type pfm: list[list[int]]
    :return: The most likely base (character representation) at each position
        in the motif
    :rtype: str

    :Example:

    >>> pfm = [[0.58, 0.12, 0.15, 0.16],
    ...       [0.2, 0.66, 0.07, 0.07],
    ...       [0.25, 0.18, 0.51, 0.06]]
    >>> pfm_consensus(pfm)
    'ACG'
    """
    numeric_list = [max(enumerate(row), key=lambda x: x[1])[0]
                        for row in pfm]
    bt = BaseTranslator()
    return bt.translate_int_to_char(numeric_list)