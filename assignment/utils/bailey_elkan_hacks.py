from typing import List, Union, Tuple, Optional
import logging
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def row_window_maxes(row_window_sums_list, motif_length):
    """
    Calculate the maximum sum of values for a sliding window of size
        `motif_length` along a row of posterior probabilities.

    :param posteriors_row: A list of posterior probabilities for a single
        sequence.
    :type posteriors_row: list[float]
    :param motif_length: The length of the motif, which sets the window size.
    :type motif_length: int

    :return: A list of maximum sums for each index along the row, considering
        a window of size `motif_length`.
    :rtype: list[float]

    :Example:

    >>> row_window_maxes([4., 7., 5., 3.], 2)
    [7.0, 7.0, 5.0, 3.0]
    """
    row_length = len(row_window_sums_list)
    max_value = 0.0
    max_values = [0.0]*row_length

    # note that this is done in reverse in order to match the results of the
    # mathematica implementation where the last window is the final value in
    # the list. a question: why aren't the windows always len(motif_length)?
    # eg [max([0.4,0.7]), max([0.7,0.5]), max([0.5,0.3])]
    for i in reversed(range(row_length)):
        if row_window_sums_list[i] > max_value:
            # row_window_sums_list[i] is not necessarily the max of the
            # new window.
            # But, if row_window_sums_list[i + motif_length] != max
            # then you know that whatever was the max in the old window is max
            # in the new window
            max_value = row_window_sums_list[i]

        # Check the value that is about to be removed from the window and
        # recalculate max_value if necessary
        # note strictly less than b/c python is 0-indexed
        if i + motif_length < row_length:
            # check to see if the max value is the one about to be removed
            if max_value == row_window_sums_list[i + motif_length]:
                # if it is, recalculate the max value in the window
                max_value = max(row_window_sums_list[i:(i + motif_length)])

        max_values[i] = max_value

    return max_values


def row_window_sums(posteriors_row, motif_length):
    """
    Calculate the running sums of a list of posterior probabilities for a
        given window size.

    :param posteriors_row: A list of posterior probabilities for a single
        sequence.
    :type posteriors_row: list[float]
    :param motif_length: The length of the motif, which sets the window size
        for calculating running sums.
    :type motif_length: int

    :return: A list of running sums along the row.
    :rtype: list[float]

    :Example:

    >>> row_window_sums([0.3, 0.5, 0.2, 0.7, 0.1], 2)
    [0.3, 0.8, 0.7, 0.8999999999999999, 0.7999999999999998]
    """
    # Initialize sum to zero
    running_sum = 0.0
    running_sums = []

    # Iterate over the row to calculate running sums
    for i, value in enumerate(posteriors_row):
        running_sum += value
        if i >= motif_length:
            running_sum -= posteriors_row[i - motif_length]
        running_sums.append(running_sum)

    return running_sums


def normalize_posterior_row(posteriors_row: np.ndarray[float],
                            motif_length: int) -> NDArray[np.float64]:
    """
    Normalize a row of posterior probabilities using the Bailey elkan
        normalization.

    :param posteriors_row: A list of posterior probabilities for a single
        sequence.
    :type posteriors_row: list[float]
    :param motif_length: The length of the motif, which sets the window size
        for normalization.
    :type motif_length: int

    :return: A list of normalized posterior probabilities.
    :rtype: list[float]

    :Example:

    >>> normalize_posterior_row([0.4, 0.7, 0.5], 2)
    [0.36363636363636365, 0.5833333333333333, 0.41666666666666663]
    """
    # Calculate the max sums for a window of size 'motif_length'
    max_sums = row_window_maxes(
        row_window_sums(posteriors_row, motif_length), motif_length
    )

    # Normalize posteriors based on calculated max sums
    return np.array(
    [
        (posterior / max_sum_value if max_sum_value > 1.0 else posterior)
        for posterior, max_sum_value in zip(posteriors_row, max_sums)
    ])


def update_eraser_row(erasers_row: NDArray,
                      posteriors_row: NDArray,
                      motif_length: int) -> NDArray:
    """
    Update an erasers row based on a given posteriors row and motif length.

    The length of an `erasers_row` is assumed to be the same as the length
    of the input row and should be greater than the length of the
    `posteriors_row` by `motif_length - 1`.

    To optimize performance, a running product is maintained and updated at
    each step. This running product is multiplied by new elements from
    `posteriors_row` and divided by old elements. If a division by zero would
    occur (i.e., the current product has a zero factor), the running product
    is reset and recalculated from scratch whenever the zero factor drops out
    of the window.

    :param erasers_row: A list of lists where each sublist is an 'eraser' for
        the corresponding sequence in the input data.
    :type erasers_row: list[list[float]]ornp.ndarray
    :param posteriors_row: A list of lists where each sublist is a list of
        posterior probabilities of a given motif for the corresponding
        sequence in the input data.
    :type posteriors_row: list[list[float]]ornp.ndarray
    :param motif_length: The length of the motif
    :type motif_length: int

    :return: A list of updated erasers.
    :rtype: list[list[float]]or np.ndarray
    """

    posteriors_length = len(posteriors_row)
    product = 1.0

    updated_erasers_row = deepcopy(erasers_row)

    for i, _ in enumerate(erasers_row):
        if i < posteriors_length:
            product *= (1.0 - posteriors_row[i])
        if i >= motif_length:
            if posteriors_row[i - motif_length] != 1.0:
                product /= (1.0 - posteriors_row[i - motif_length])
            else:
                # If the factor you're trying to divide out was zero,
                # then you have to recalculate the product for the entire
                # new window.
                product = 1.0
                for j in range(i - motif_length + 1,
                                min(i + 1, posteriors_length)):
                    product *= (1.0 - posteriors_row[j])

        updated_erasers_row[i] *= product

    return updated_erasers_row