import unittest
import numpy as np
from .BaseTranslator import BaseTranslator
from .bailey_elkan_hacks import (row_window_sums,
                                 row_window_maxes,
                                 normalize_posterior_row,
                                 update_eraser_row)

class TestSiteEMUtilities(unittest.TestCase):

    def test_BaseTranslator(self):
        bt = BaseTranslator()
        sequences = ['AATA', 'GAAAA', 'TATAT']
        sequences_int_list = [bt.translate_char_to_int(seq)
                             for seq in sequences]

        actual = [bt.translate_int_to_char(seq) for seq in sequences_int_list]
        self.assertEqual(actual, sequences)
    
    def test_calculate_running_sums_for_row(self):
        row = [1., 2., 3., 4., 5.]
        motif_length = 2
        result = row_window_sums(row, motif_length)
        self.assertEqual(result, [1, 3, 5, 7, 9])

    def test_calculate_max_sums_for_row(self):
        row = [4., 7., 5., 3.]
        motif_length = 2
        result = row_window_maxes(row, motif_length=2)
        self.assertEqual(result, [7., 7., 5., 3.])

    def test_normalize_posterior_row(self):
        row = [.3, .6, .9]
        motif_length = 2
        result = normalize_posterior_row(row, motif_length)
        self.assertAlmostEqual(sum(np.array([0.3, 0.4, 0.6]) - result), 
                               0, 
                               places = 10),

    def test_update_eraser_row(self):
        row = [.1, .2, .3, .4, .5]
        motif_length = 2
        erasers_1 = [1., 1., 1., 1., 1., 1.]
        erasers_2 = [1., 1., 0.5, 1., 1., 0.5]
        result_1 = update_eraser_row(erasers_1, row, motif_length)
        result_2 = update_eraser_row(erasers_2, row, motif_length)
        for a, b in zip(result_1, [0.9, 0.72, 0.56, 0.42, 0.3, 0.5]):
            self.assertAlmostEqual(a, b, places=10)

        for a, b in zip(result_2, [0.9, 0.72, 0.28, 0.42, 0.3, 0.25]):
            self.assertAlmostEqual(a, b, places=10)

    def test_update_erasers(self):
        erasers = [[1.0, 1.0, 0.5, 1.0, 1.0, 0.5],
                   [1.0, 1.0, 1.0, 1.0, 1.0]]
        rows = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 1.0, 0.4]]
        motif_length = 2
        result = [update_eraser_row(a, b, motif_length)
                  for a, b in zip(erasers, rows)]
        expected = [[0.9, 0.72, 0.28, 0.42, 0.3, 0.25],
                    [0.9, 0.72, 0., 0., 0.6]]
        for sublist1, sublist2 in zip(result, expected):
            for a, b in zip(sublist1, sublist2):
                self.assertAlmostEqual(a, b, places=10)