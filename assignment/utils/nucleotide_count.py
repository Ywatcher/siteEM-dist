import numpy as np
from typing import List


def nucleotide_count(
        sequences: List[List[int]]) -> np.ndarray:
    """
    Calculate the base frequencies of a set of sequences

    :param sequences: A list lists where each internal list represents a
        sequence of integers representing bases. The integers should be
        0, 1, 2, or 3, representing A, C, G, or T, respectively.
    :type sequences: list[list[int]] or np.ndarray

    :return: A numpy array of the base frequencies in the input set of
        sequences
    :rtype: np.ndarray

    :Example:

    >>> sequences = [[0, 1, 2, 3], [0, 0, 0, 0]]
    >>> nucleotide_count(sequences)
    array([5, 1, 1, 1])
    """

    flattened = np.concatenate([np.array(sublist) for sublist in sequences])

    # Initialize an array for frequency counting
    count_array = np.zeros(4, dtype=int)

    # np.unique returns a tuple where the first element is the set of unique
    # values and the second element is the number of times each unique value
    # appears in the array
    unique, counts = np.unique(flattened, return_counts=True)
    # Add the counts to the array. Using `unique` adds the counts' values to
    # the correct indicies
    count_array[unique] = counts

    return count_array