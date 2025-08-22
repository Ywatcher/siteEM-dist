import logging
import unittest
import numpy as np
from importlib.resources import path
from copy import deepcopy
from gradescope_utils.autograder_utils.decorators import weight # type:ignore
from cse587Autils.configure_logging import configure_logging
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
from .utils.consensus_pfm import consensus_pfm
from .assignment import (e_step,
                       update_motif_prior,
                       update_site_probs,
                       m_step,
                       siteEM,
                       siteEM_intializer)

configure_logging(logging.INFO)


class TestInternals(unittest.TestCase):

    def setUp(self) -> None:
        site_base_probs = [[.2, .2, .3, .3],
                           [.2, .3, .2, .3]]
        background_base_probs = [1/4]*4
        site_prior = 1/2
        self.sequence_model = SequenceModel(
            site_prior, site_base_probs, background_base_probs)

    @weight(2)
    def test_e_step_1(self):
        """Test sequencePosteriors in a simple case where all should be
            equal
        """
        sequence = [0] * 5

        result = e_step(sequence, self.sequence_model, bailey_elkan_norm=False)

        for a, b in zip(result, [0.39, 0.39, 0.39, 0.39]):
            self.assertAlmostEqual(a, b, places=2)

    @weight(2)
    def test_e_step_2(self):
        """Test sequencePosteriors to make sure relative posteriors match
            intuition
        """
        sequence = [3, 3, 2, 3, 1, 0]

        result = e_step(sequence, self.sequence_model, bailey_elkan_norm=False)

        for a, b in zip(result, [0.59, 0.49, 0.59, 0.59, 0.39]):
            self.assertAlmostEqual(a, b, places=2)

    @weight(2)
    def test_e_step_3(self):
        """Make sure that the posteriors respond to changes in the prior"""
        sequence_model_update = deepcopy(self.sequence_model)
        sequence_model_update.site_prior = 1/4
        sequence = [3, 3, 2, 3, 1, 0]

        result = e_step(sequence, sequence_model_update,
                        bailey_elkan_norm=False)

        for a, b in zip(result, [0.32, 0.24, 0.32, 0.32, 0.18]):
            self.assertAlmostEqual(a, b, places=2)

    @weight(2)
    def test_e_step_4(self):
        """Test the case where the sequence is the same length as the
            motif/PFM"""
        sequence_model_update = deepcopy(self.sequence_model)
        sequence_model_update.site_prior = 1/4
        sequence = [0] * 2

        result = e_step(sequence, sequence_model_update,
                        bailey_elkan_norm=False)

        for a, b in zip(result, [0.18]):
            self.assertAlmostEqual(a, b, places=2)

    @weight(2)
    def test_update_motif_prior_1(self):
        """Test the update_motif_prior function"""

        sequences = [[0]*5, [3, 3, 2, 3, 1, 0]]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=False)
                      for seq in sequences]
        result = update_motif_prior(posteriors)
        # note, the unrounded value is: 0.467945248204573
        # using the Bailey-elkan normalization, the result is:
        # 0.43902439024390244
        self.assertAlmostEqual(result, 0.47, places=2)

    @weight(2)
    def test_update_motif_prior_2(self):
        """Make sure the new prior responds to a change in the old prior"""
        sequence = [[0]*5, [3, 3, 2, 3, 1, 0]]
        sequence_model_update = deepcopy(self.sequence_model)
        sequence_model_update.site_prior = 1/4

        posteriors = [e_step(seq, sequence_model_update, bailey_elkan_norm=False)
                      for seq in sequence]
        result = update_motif_prior(posteriors)

        # note, the unrounded value is: 0.23272423272423273
        # using the Bailey-elkan normalization, the result is the same
        self.assertAlmostEqual(result, 0.23, places=2)

    @weight(2)
    def test_update_motif_prior_3(self):
        """
        Change the sequence to be a better match for the PFM.
            The new prior should go up
        """
        sequences = [[3, 2, 3, 1, 2], [3, 3, 2, 3, 1, 0]]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=False)
                      for seq in sequences]
        result = update_motif_prior(posteriors)
        # note, the unrounded value is: 0.5234332570826544
        # using the Bailey-elkan normalization, the result is:
        # 0.47044660037678443
        self.assertAlmostEqual(result, 0.52, places=2)

    @weight(2)
    def test_update_site_probs_1(self):
        """
        If one sequence is a better match to the motif than another, the
        motif posterior is higher for that sequence, hence it's
        nucleotides are weighted more heavily in the expected counts.
        For nucleotides that don' t occur at all, their new count is the
        pseudocount, which here is set to be 0.25 each.
        """
        # this has a "Snip" note in the mathematic code?
        sequences = [[0, 0], [3, 3]]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=True)
                      for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[1]*len(x) for x in sequences]

        result = update_site_probs(sequences,
                                   len(self.sequence_model.site_base_probs),
                                   posteriors,
                                   motif_pseudocounts,
                                   erasers,
                                   normalize=False)
        expected = [[0.64, 0.25, 0.25, 0.84], [0.64, 0.25, 0.25, 0.84]]
        for a, b in zip(result, expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_update_site_probs_2(self):
        """
        1 never occurs in the first column so it' s count is just the
        pseudocount, 0.25. The biggest expected count is 4 in the first
        column, because it occurs frequently in the first column and when
        it does, the sequence tends to be a good fit for the motif, so the
        posterior is high. 4 occurs less frequently in the second column
        than in the first, because two of its occurrence are at the
        beginnings of sequences
        """
        sequences = [[3, 2, 3, 1, 2],
                     [3, 3, 2, 3, 1, 0]]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=False)
                      for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[1]*len(x) for x in sequences]

        result = update_site_probs(sequences,
                                   len(self.sequence_model.site_base_probs),
                                   posteriors,
                                   motif_pseudocounts,
                                   erasers,
                                   normalize=False)
        expected = [[0.25, 1.03, 1.43, 3.],
                    [0.64, 1.43, 1.62, 2.02]]

        for a, b in zip(result, expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_update_site_probs_3(self):
        """
        providing False for the erasers should have the same result
        """
        sequences = [[3, 2, 3, 1, 2],
                     [3, 3, 2, 3, 1, 0]]
        erasers = [np.ones(len(seq)) for seq in sequences]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=False)
                      for seq in sequences]
        motif_pseudocounts = [1/4]*4

        result = update_site_probs(sequences,
                                   len(self.sequence_model.site_base_probs),
                                   posteriors,
                                   motif_pseudocounts,
                                   erasers,
                                   normalize=False)

        expected = [[0.25, 1.03, 1.43, 3.],
                    [0.64, 1.43, 1.62, 2.02]]

        for a, b in zip(result, expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_update_site_probs_4(self):
        """
        Making the erasers smaller for all the input sites with 4' s should
        reduce the expected counts for 4. Expected counts for other bases
        are unaffected since the erasers are applied after the posteriors
        are computed
        """
        sequences = [[3, 2, 3, 1, 2],
                     [3, 3, 2, 3, 1, 0]]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=False)
                      for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[0.5, 1, 0.5, 1, 1],
                   [0.5, 0.5, 1, 0.5, 1, 1]]

        result = update_site_probs(sequences,
                                   len(self.sequence_model.site_base_probs),
                                   posteriors,
                                   motif_pseudocounts,
                                   erasers,
                                   normalize=False)

        expected = [[0.25, 1.03, 1.43, 1.63],
                    [0.64, 1.43, 1.62, 1.14]]

        for a, b in zip(result, expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_test_m_step_1(self):
        """
        Updated PFM probabilites must sum to 1 (or very close) in each column.
        """
        sequences = [[0, 0],
                     [3, 3]]
        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=True)
                      for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[1]*len(x) for x in sequences]

        motif_prior_actual, motif_pfm_actual = \
            m_step(sequences,
                   len(self.sequence_model.site_base_probs),
                   posteriors,
                   motif_pseudocounts,
                   erasers)
        summed_result = [sum(x) for x in motif_pfm_actual]
        expected = [1, 1]

        for a, b in zip(summed_result, expected):
            self.assertAlmostEqual(a, b, places=6)

    @weight(2)
    def test_test_m_step_2(self):
        """
        On this simple input which includes only 0's and 3's, the update
        should increase the probabilities of 0's and 3's at the expense of
        1's and 2's
        """
        # This call to m_step is the same as the previouos one but the 
        # assert is different.
        sequences = [[0, 0],
                     [3, 3]]

        posteriors = [e_step(seq, self.sequence_model, bailey_elkan_norm=True)
                      for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[1]*len(x) for x in sequences]

        motif_prior_actual, motif_pfm_actual = \
            m_step(sequences,
                   len(self.sequence_model.site_base_probs),
                   posteriors,
                   motif_pseudocounts,
                   erasers)

        motif_prior_expected = 0.49
        self.assertAlmostEqual(motif_prior_actual,
                               motif_prior_expected,
                               places=2)

        motif_pfm_expected = [[0.32, 0.13, 0.13, 0.42],
                              [0.32, 0.13, 0.13, 0.42]]
        for a, b in zip(motif_pfm_actual, motif_pfm_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_test_m_step_3(self):
        """
        This update should increase the probabilities assigned to 3 in the
        first column and 2 in the second column.
        """
        sequences = [[3, 2, 3, 1, 2],
                     [3, 3, 2, 3, 1, 0]]
        posteriors_actual = \
            [e_step(seq, self.sequence_model, bailey_elkan_norm=True)
             for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[1]*len(x) for x in sequences]

        motif_prior_actual, motif_pfm_actual = \
            m_step(sequences,
                   len(self.sequence_model.site_base_probs),
                   posteriors_actual,
                   motif_pseudocounts,
                   erasers)

        motif_prior_expected = 0.47
        self.assertAlmostEqual(motif_prior_actual,
                               motif_prior_expected,
                               places=2)

        posteriors_expected = [[0.45, 0.5, 0.5, 0.39],
                               [0.55, 0.45, 0.5, 0.5, 0.39]]
        for a, b in zip(posteriors_actual, posteriors_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

        motif_pfm_expected = [[0.05, 0.2, 0.24, 0.52],
                              [0.12, 0.24, 0.3, 0.34]]
        for a, b in zip(motif_pfm_actual, motif_pfm_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_test_m_step_4(self):
        """
        Making the erasers smaller for all the input sites with 4' s should
        reduce the updated PFM probabilities for 4' s, relative to no erasers.
        This will raise the updated PFM probabilities for 1 - 3, since
        probabilies must sum to one. The posteriors should not be affected.
        """
        sequences = [[3, 2, 3, 1, 2],
                     [3, 3, 2, 3, 1, 0]]

        posteriors_actual = \
            [e_step(seq, self.sequence_model, bailey_elkan_norm=True)
                for seq in sequences]
        motif_pseudocounts = [1/4]*4
        erasers = [[0.5, 1, 0.5, 1, 1],
                   [0.5, 0.5, 1, 0.5, 1, 1]]

        motif_prior_actual, motif_pfm_actual = \
            m_step(sequences,
                   len(self.sequence_model.site_base_probs),
                   posteriors_actual,
                   motif_pseudocounts,
                   erasers)

        motif_prior_expected = 0.47
        self.assertAlmostEqual(motif_prior_actual,
                               motif_prior_expected,
                               places=2)

        posteriors_expected = [[0.45, 0.5, 0.5, 0.39],
                               [0.55, 0.45, 0.5, 0.5, 0.39]]
        for a, b in zip(posteriors_actual, posteriors_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

        motif_pfm_expected = [[0.06, 0.26, 0.31, 0.37],
                              [0.14, 0.28, 0.35, 0.23]]
        for a, b in zip(motif_pfm_actual, motif_pfm_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)


class TestSiteEM(unittest.TestCase):

    def setUp(self):
        # set up a sequence model. This may be copied and updated in the tests
        self.sequence_model = SequenceModel(
            site_prior=1/5,
            site_base_probs=[[.2, .2, .3, .3], [.2, .3, .2, .3]],
            background_base_probs=[1/4]*4)

    @weight(1)
    def test_1(self):
        """
        If the input consists entirely of a single nucleotide, the
        inferred motif should assign that nucleotide a high probability
        in all columns
        """
        result = siteEM(
            [[0,0,0,0,0], [0, 0, 0, 0]],
            self.sequence_model,
            self.sequence_model.background_base_probs,
            max_iterations=1)

        site_probs_expected = [[0.62, 0.13, 0.13, 0.13],
                               [0.62, 0.13, 0.13, 0.13]]

        for a, b in zip(result[0][1].site_base_probs, site_probs_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(1)
    def test_2(self):
        """
        More iterations should further increase the probability of the
        single nucleotide in the input, up to some limit determined by
        pseudocounts.
        """
        result = siteEM(
            [[0,0,0,0,0], [0, 0, 0, 0]],
            self.sequence_model,
            self.sequence_model.background_base_probs,
            max_iterations=10,
            accuracy=0.001)

        site_probs_expected = [[0.83, 0.06, 0.06, 0.06],
                               [0.83, 0.06, 0.06, 0.06]]

        for a, b in zip(result[0][1].site_base_probs, site_probs_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_3(self):
        """
        The window posteriors should high, since the leraned motif is a good
        fit for the data everywhere in the sequence. However, window
        normalization will prevent any two consecutive posteriors from summing
        to more than 1 -- with this input, they will be 0.5
        """
        result = siteEM(
            [[0,0,0,0,0], [0, 0, 0, 0]],
            self.sequence_model,
            self.sequence_model.background_base_probs,
            max_iterations=10,
            accuracy=0.001)

        window_posteriors_expected = [[0.5, 0.5, 0.5, 0.5],
                                      [0.5, 0.5, 0.5]]

        for a, b in zip(result[0][0], window_posteriors_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_4(self):
        """
        With a motif of length 3, window normalization will prevent any 3
        consecutive posteriors from summing to more than 1.
        """
        sequence_model_update = deepcopy(self.sequence_model)
        sequence_model_update.site_base_probs = [[.2, .2, .3, .3],
                                                 [.2, .3, .2, .3],
                                                 [.3, .3, .2, .2]]

        result = siteEM(
            [[0,0,0,0,0], [0, 0, 0, 0]],
            sequence_model_update,
            sequence_model_update.background_base_probs,
            max_iterations=10,
            accuracy=0.001)

        window_posteriors_expected = [[0.33, 0.33, 0.33],
                                      [0.5, 0.5]]

        for a, b in zip(result[0][0], window_posteriors_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_5(self):
        """
        Find a repeated Pattern. First check that the final PFM favors 1 in
        the first column, 2 in the second column, and 3 in the third column
        """
        sequence_model_update = deepcopy(self.sequence_model)
        sequence_model_update.site_base_probs = [[.2, .2, .3, .3],
                                                 [.2, .3, .2, .3],
                                                 [.3, .3, .2, .2]]

        result = siteEM(
            [[3, 0, 1, 2, 0],
             [2, 1, 0, 0, 1, 2, 3, 1, 0, 1, 2]],
            sequence_model_update,
            sequence_model_update.background_base_probs,
            max_iterations=10,
            accuracy=0.001)

        motif_pfm_expected = [[0.58, 0.12, 0.15, 0.16],
                              [0.2, 0.66, 0.07, 0.07],
                              [0.25, 0.18, 0.51, 0.06]]

        for a, b in zip(result[0][1].site_base_probs, motif_pfm_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(2)
    def test_6(self):
        """_
        Find a repeated Pattern. Now check that the posteriors have the highest
        probability in the positions where the pattern 012 occurs.
        """
        sequence_model_update = deepcopy(self.sequence_model)
        sequence_model_update.site_base_probs = [[.2, .2, .3, .3],
                                                 [.2, .3, .2, .3],
                                                 [.3, .3, .2, .2]]

        result = siteEM(
            [[3, 0, 1, 2, 0],
             [2, 1, 0, 0, 1, 2, 3, 1, 0, 1, 2]],
            sequence_model_update,
            sequence_model_update.background_base_probs,
            max_iterations=10,
            accuracy=0.001)

        window_posteriors_expected = [[0.15, 0.8, 0.05],
                                      [0.38, 0.11, 0.28, 0.61, 0.01, 0.05,
                                       0.3, 0.09, 0.61]]

        for a, b in zip(result[0][0], window_posteriors_expected):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)


class TestSiteEMInitializer(unittest.TestCase):
    def setUp(self) -> None:
        # read in fasta files using file-based paths
        import os
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.tiny_fasta_path = os.path.join(data_dir, 'tinyTest.fasta')
        self.small_fasta_path = os.path.join(data_dir, 'smallTest.fasta')  
        self.pac_fasta_path = os.path.join(data_dir, 'PACPlusSeqs.fasta')
    @weight(2)
    def test_siteEM_initializer_1(self):
        """
        Test on the simplest of the input files, with motiflength 2.
        TA occurs three times and AA occurs four times, so it should find
        one of those.
        """
        results = siteEM_intializer(
            self.tiny_fasta_path, 2, 0.01,
            max_iterations=100,
            accuracy=0.01,
            site_base_probs_seed=42)

        consensus_motif_expected = {"AA"}
        self.assertTrue(consensus_pfm(results[0][1].site_base_probs)
                        in consensus_motif_expected)

        pfm_expected = [[0.62, 0.0, 0.08, 0.30],
                        [0.75, 0.0, 0.00, 0.25]]
        # Zip together the site_base_probs so a and b are now single rows
        for a, b in zip(results[0][1].site_base_probs, pfm_expected):
            # Zip together the contents of each row so c, d are probs for same base
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d, places=2)

    @weight(3)
    def test_siteEM_initializer_2(self):
        """
        Test on the mid size file, still artificially generated.
        Here we will only focus on testing the consenus
        """
        results = siteEM_intializer(
            self.small_fasta_path, 4, 0.01,
            max_iterations=100,
            accuracy=0.001,
            site_base_probs_seed=40)

        consensus_motif_expected = "GTGA"
        self.assertEqual(consensus_pfm(results[0][1].site_base_probs),
                         consensus_motif_expected)

    @weight(5)
    def test_siteEM_initializer_3(self):
        """
        Test multiple motifs and erasers on mid sized file, focussing on
        testing the consenu
        """
        results = siteEM_intializer(
            self.small_fasta_path, 4, 0.01,
            max_iterations=1000,
            accuracy=0.001,
            num_motifs_to_find=3,
            site_base_probs_seed=42)
        
        # NOTE: original mathematica test had the second motif as TCAC.
        # I inputed smallFasta to the meme suite with comparable settings
        # (motif length 4, 3 motifs, only input strand, any 
        # number of motifs per sequence) and CACA is as frequently found as
        # GTGA
        consensus_motif_expected = ["GTGA", "CACA", "TTTT"]
        for a, b in zip([consensus_pfm(x[1].site_base_probs) for x in results],
                        consensus_motif_expected):
            self.assertEqual(a, b)
