# pylint: disable=C0103
from typing import List, Union, Tuple, Optional
import logging
from math import inf
from copy import deepcopy
from pathlib import PosixPath
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
import numpy as np
from numpy.typing import NDArray
from .utils.bailey_elkan_hacks import \
    normalize_posterior_row, update_eraser_row
from .utils.read_in_fasta import read_in_fasta
from .utils.nucleotide_count import nucleotide_count

logger = logging.getLogger(__name__)

# e_step takes a single input sequence, not the list of all input sequences.
def e_step(sequence: List[int],
           sequence_model: SequenceModel,
           bailey_elkan_norm: bool = True) -> NDArray[np.float64]:
    """
    Perform the E step of the EM algorithm. In the context of TF motif
        inference, this calculates the posterior probability of a bound site
        versus an unbound site for each position in a sequence. Note that
        the posteriors are normalized per Bailey and elkan 1993.

    :param sequence: Observed bases represented as integers where 0 = A,
        1 = C, 2 = G and 3 = T
    :param sequence_model: A SequenceModel object which stores the parameters
        of the genome model which produced the sequence
    :param bailey_elkan_norm: If True, normalize the posteriors per Bailey and
        elkan 1993. If False, do not normalize the posteriors.
    :return: Posterior probability of a bound site for each position in the
        sequence normalized per Bailey and elkan 1993
    :raises ValueError: If len(sequence) is less than len(sequence_model)
    """
    if not len(sequence) >= len(sequence_model):
        raise ValueError("sequence must be longer than site_base_probs")
    if not isinstance(sequence_model, SequenceModel):
        raise TypeError("sequence_model must be a SequenceModel object")
    
    # posteriors_row is the NDArray of posteriors for a single input sequence.
    posteriors_row = np.zeros(len(sequence) - sequence_model.motif_length() + 1)

    # iterate a sliding window of length motif_length over the sequence
    # eg. for motif length 2 and sequence [0, 1, 2, 3], the
    # windows are [0, 1], [1, 2], [2, 3]. Your code will call site_posterior on 
    # each of these windows and store the result in posteriors_row.
    # remember that python is 0 indexed and exclusive of the end index
    # <snip>
    for posterior_index, _ in enumerate(posteriors_row):
        subseq = sequence[posterior_index:
                          posterior_index+sequence_model.motif_length()]
        posteriors_row[posterior_index] = site_posterior(subseq,
                                                          sequence_model)
    # </snip>
    # return the normalized posteriors per Bailey and elkan 1993
    return normalize_posterior_row(posteriors_row,
                                   sequence_model.motif_length()) \
        if bailey_elkan_norm else posteriors_row


def update_motif_prior(posteriors: List[NDArray[np.float64]]) -> float:
    """
    Given a list of posterior probabilities, calculate the prior probability of
        observing a motif vs. background at any given position in the sequence.

    :param posteriors: a list of lists. Each sublist contains the posterior
        probabilities of a subsequence length motif_length within
        a given input sequence being bound
    :type posteriors: list[list[float]]

    :raises ValueError: If the expected motif count exceeds the maximum motif
        count

    :return: The probability of observing a motif
    :rtype: float
    """
    # Calculate the expected total motif count, summed over all sequences. 
    # This is the sum of the posteriors. Then normalize by the maximum possible
    # motif count to get the udpated prior.
    # <snip>
    expected_motif_count = sum([sum(row) for row in posteriors])
    max_motif_count = sum([len(row) for row in posteriors])

    return expected_motif_count / max_motif_count
    # </snip>

def update_site_probs(sequences:  List[List[int]],
                      motif_length: int,
                      posteriors: List[NDArray[np.float64]],
                      motif_pseudocounts: List[float],
                      erasers: List[NDArray[np.float64]],
                      normalize: bool = True) -> List[List[float]]:
    """
    Update the site probabilities based on the posteriors and the sequence.

    :param sequences: A list of lists. Each sublist contains the bases of a
    :type sequences: list[list[int]]
    :param motif_length: number of positions (bases) in the motif
    :type motif_length: int
    :param posteriors: a list of ndarrays. Each sublist contains the posterior
        probabilities of a subsequence length motif_length within
        a given input sequence being bound
    :type posteriors: list[NDArray[float]]
    :param motif_pseudocounts: A list of 4 floats representing the pseudocount
        for each base
    :type motif_pseudocounts: lit[float]
    :param erasers: A list of lists. Each sublist contains the eraser
        probabilities for a subsequence length motif_length within
        a given input sequence being bound. If False, do not use erasers.
    :type erasers: list[list[float]] or False
    :param normalize: Controls whether to return the normalized or unnormalized
        site probabilities. Defaults to True
    :type normalize: bool, optional

    :return: A list of lists. Each sublist contains th probabilities of
        observing each of the 4 bases at a given position in the motif
    :rtype: list[list[float]]
    """
    # instantiate a list of lists to hold the unnormalized site probs.
    unnormalized_site_probs = [deepcopy(motif_pseudocounts)
                               for _ in range(motif_length)]
    # update the pfm by looping over sequences, windows within each sequence,
    # and positions within each window. When calculating the expected frequency
    # of a given letter in a given position of the motif, multiply posteriors by 
    # erasers. Note the lenghth of the erasers is that of the sequence, not that
    # of the posteriors -- you'll need to index into the erasers list to get the
    # correct eraser for the current position in the motif. You will need 3 nested
    # loops to do this.
    # <snip>
    # the outer loop iterates over the sequences
    for i, seq in enumerate(sequences):
        # extract the posterior vector for the current sequence. The values in
        # each entry in posterior corresponds to the posterior probability of
        # a window of length motif_length being bound
        posteriors_row = posteriors[i]
        # the second level loop iterates over windows of length motif_length
        # in the current sequence
        for j in range(len(seq) - motif_length + 1):
            # extract the subsequence of length motif_length from the current
            # sequence
            subseq = seq[j:j+motif_length]
            # the inner loop indexes the bases in the subsequence and
            # positions in the motif.
            for k, base in enumerate(subseq):
                unnormalized_site_probs[k][base] += \
                    (posteriors_row[j] * erasers[i][j + k])

    # </snip>
    # calculate the sum of each position in the pfm
    normalizer = np.sum(unnormalized_site_probs, axis=1)[:, np.newaxis]

    return (unnormalized_site_probs / normalizer) if normalize \
        else unnormalized_site_probs


def m_step(sequences: List[List[int]],
           motif_length: int,
           posteriors: List[NDArray[np.float64]],
           motif_pseudocounts: List[float],
           erasers: List[NDArray[np.float64]]) \
            -> Tuple[float, List[List[float]]]:
    """
    Update the motif prior and site probabilities based on the posteriors

    :param sequences: A list of lists. Each sublist contains the bases of a
        sequence with numeric bases represented as integers where 0 = A,
        1 = C, 2 = G and 3 = T
    :type sequences: list[list[int]]
    :param motif_length: number of positions (bases) in the motif
    :type motif_length: int
    :param posteriors: a list of numpy arrays. Each array contains the posterior
        probabilities of a subsequence of length motif_length within
        a given input sequence being generated by the motif model.
    :type posteriors: List[NDAarray[float]]
    :param motif_pseudocounts: A list of 4 floats representing the pseudocount
        for each base
    :type motif_pseudocounts: list[float]
    :param erasers: A list of lists. Each sublist contains the eraser
        probabilities for a subsequence length motif_length within
        a given input sequence being bound. If False, do not use erasers.
    :type erasers: List[NDArray[np.float64]]

    :return: A tuple containing the updated motif prior and site probabilities
    :rtype: tuple[list[float], list[list[float]]]
    """
    # Update the motif prior and site probabilities based on the posteriors.
    # <snip>
    updated_site_prior = update_motif_prior(posteriors)

    updated_site_probs = update_site_probs(sequences,
                                           motif_length,
                                           posteriors,
                                           motif_pseudocounts,
                                           erasers)
    # </snip>
    return updated_site_prior, updated_site_probs


def siteEM(sequences: List[List[int]],
           init_sequence_model: SequenceModel,
           motif_pseudocounts: List[float],
           max_iterations: int = 1000,
           accuracy: float = 1e-6,
           num_motifs_to_find: int = 1) \
            -> List[Tuple[List[NDArray[np.float64]],SequenceModel]]:
    """
    This is the main function which conducts the MEME EM algorithm. It
        iterates through the E and M steps until the parameters converge or
        the maximum number of iterations is reached. If more than one motif
        is to be found, the erasers are updated after each iteration.

    :param sequences: A list of lists. Each sublist contains the bases of a
        sequence with numeric bases represented as integers where 0 = A,
        1 = C, 2 = G and 3 = T
    :type sequences: np.ndarray[np.ndarray[int]]
    :param init_sequence_model: A SequenceModel object which stores the
        parameters of the genome model which produced the sequence
    :type init_sequence_model: SequenceModel
    :param motif_pseudocounts: A list of 4 floats representing the pseudocount
        for each base
    :type motif_pseudocounts: list[list[float]]
    :param max_iterations: The maximum number of iterations which the EM
        algorithm may make. Defaults to 1000
    :type max_iterations: int, optional
    :param accuracy: If the euclidean distance between the updated parameters
        in a given iteration and the previous parameters is less than
        this value, the EM algorithm will halt and report the current
        updated parameters as the result. Defaults to 1e-6
    :type accuracy: float, optional
    :param num_motifs_to_find: The number of motifs to search for in the
        sequences. Defaults to 1
    :type num_motifs_to_find: int, optional
    :return: A tuple containing the normalized posteriors and the updated
        set of sequence model parameters.
    :rtype: List[Tuple[List[List[float]], SequenceModel]]
    """

    # initialize the erasers
    erasers = [np.ones(len(seq)) for seq in sequences]
    
    # Perform EM iterations
    for index in range(num_motifs_to_find):
        iteration = 0
        parameter_diff = inf
        current_sequence_model = init_sequence_model
        while parameter_diff > accuracy and iteration < max_iterations:
            # E-step
            # Note: the E-step is run on each sequence in the input list of
            # sequences
            # <snip>
            normalized_posteriors = [
                e_step(seq, current_sequence_model) for seq in sequences]
            # </snip>
            # M-step. The M-step requires the entire list of sequences.
            # <snip>
            updated_site_prior, updated_site_probs = \
                m_step(sequences,
                       len(current_sequence_model),
                       normalized_posteriors,
                       motif_pseudocounts,
                       erasers=erasers)
            # </snip>

            # set the prev_sequence_model to a copy of the parameters which
            # where used in this iteration

            prev_sequence_model = current_sequence_model
            # store the updated parameters in a SequenceModel object.
            # the background probs do not change between iterations
            current_sequence_model = SequenceModel(
                updated_site_prior,
                updated_site_probs,
                init_sequence_model.background_base_probs)
            # calculate the amount that the parameters changed. If he amount
            # of change is less than `accuracy`, the while loop will terminat
            parameter_diff = current_sequence_model - prev_sequence_model
            # increment the iteration counter
            iteration += 1
            # print("Iteration ", iteration, "parmeter_diff", parameter_diff)
        # return a tuple of the normalized posteriors and the current set
        # of sequence model parameters
        if index == 0:
            results = [(normalized_posteriors, current_sequence_model)]
        else:
            results.append((normalized_posteriors, current_sequence_model))
        # Update erasers
        erasers = [update_eraser_row(erasers_row,
                                     posteriors_row,
                                     len(current_sequence_model))
                    for erasers_row, posteriors_row in
                    zip(erasers, normalized_posteriors)]
    return results


def siteEM_intializer(fasta: Tuple[str, PosixPath],
                      motif_length: int,
                      pseudocount_weight: float = 0.01,
                      **kwargs) -> List[Tuple[List[NDArray[np.float64]],SequenceModel]]:

    """
    This function reads in the fasta file, parses the alphabetical
    representation of the bases into integers, initializes an instance of
    SequenceModel to serve as the initial parameters for the EM algorithm,
    and calls the siteEM function.

    The kwargs argument allows the user to pass in optional arguments to
        the functions, including siteEM, which are called by this function.
        Currently, the following optional arguments are supported:
        - max_iterations: The maximum number of iterations which the EM
            algorithm may make. Defaults to 1000
        - accuracy: If the sum of the absolute values of differences between
            between the updated parameters
            in a given iteration and the previous parameters is less than
            this value, the EM algorithm will halt and report the current
            updated parameters as the result. Defaults to 1e-6
        - num_motifs_to_find: The number of motifs to search for in the
            sequences. Defaults to 1
        - site_base_probs_seed: The seed to use when initializing the
            site_base_probs. Defaults to 42
        - include_reverse_complement: If True, include the reverse complement
            of the motif in the results. Defaults to False

    :param fasta: Path to fasta file
    :type fasta: str, PosixPath
    :param motif_length: The length of the motif to find
    :type motif_length: int
    :param pseudocount_weight: The weight to use when calculating the
        pseudocounts. Defaults to 0.01
    :type pseudocount_weight: float

    :return: A tuple containing the normalized posteriors and the updated
        set of sequence model parameters.
    :rtype: tuple[list[list[float]], SequenceModel]
    """

    sequences = read_in_fasta(
        fasta,
        include_reverse_complement=kwargs.get('include_reverse_complement',
                                              False))
    base_counts = nucleotide_count(sequences)
    # backgroundFreqs represents the frequency with which each nucleotide
    # occurs in the promoter sequences of which the input sequences are a
    # sample, at sites not generated by the motif. These should be the vast
    # majority sites in the promoter regions. There is a strong prior that no
    # nucleotides should be excluded from such "generic" promoter sites.
    # In fact, the lowest nucleotide frequency should be within roughly a
    # factor of two of the greatest, so significant pseudocounts are justified.
    background_pseudocounts = [1]*4
    background_base_probs = ((base_counts + background_pseudocounts) /
                             (sum(base_counts) + sum(background_pseudocounts)))
    motif_pseudocounts = [pseudocount_weight * x
                          for x in background_base_probs]

    # In the published paper they iterate through many combinations of
    # initializations for the PFM and the prior, or expected number of motif
    # occurrences. They say this is important and is the primary determinant
    # of running time. Not implemented here
    # The prior probability of finding a motif at a given site is one over the
    # average length of the input sequences, so the expectatation based on this
    # prior is one motif per input. *)
    initial_motif_prior = len(sequences) / sum([len(x) for x in sequences])

    # Initialize the motif model
    site_base_probs_seed =kwargs.get('site_base_probs_seed', 42)
    np.random.seed(site_base_probs_seed)
    site_base_probs = np.random.uniform(0.5, 1.0, (motif_length, 4))

    # Sum along axis=1 to get the sum of each row
    row_sums = site_base_probs.sum(axis=1)

    # Normalize
    site_base_probs = site_base_probs / row_sums[:, np.newaxis]

    sequence_model = SequenceModel(initial_motif_prior,
                                   site_base_probs,
                                   background_base_probs)
 
    return siteEM(sequences,
                  sequence_model,
                  motif_pseudocounts,
                  max_iterations=kwargs.get('max_iterations', 1000),
                  accuracy=kwargs.get('accuracy', 1e-6),
                  num_motifs_to_find=kwargs.get('num_motifs_to_find', 1))

def site_posterior (O00OO00O0OO0OO00O :list [int ],OO00O0000OOO0O000 :SequenceModel )->float :#line:2
    ""#line:25
    if not isinstance (O00OO00O0OO0OO00O ,list ):#line:27
        raise TypeError ("sequence must be a list")#line:28
    if not len (O00OO00O0OO0OO00O )==OO00O0000OOO0O000 .motif_length ():#line:29
        raise ValueError ("sequence and site_base_probs must be the same length")#line:31
    for O0OOOOOOO0O000OOO in O00OO00O0OO0OO00O :#line:32
        if not isinstance (O0OOOOOOO0O000OOO ,int ):#line:33
            raise TypeError ("sequence must be a list of integers")#line:34
        if O0OOOOOOO0O000OOO <0 or O0OOOOOOO0O000OOO >3 :#line:35
            raise ValueError ("sequence must be a list of integers between 0 " "and 3 (inclusive)")#line:37
    O0O00OO0O00O0O00O =OO00O0000OOO0O000 .site_prior #line:41
    O0O0O0O0O0OOOO00O =OO00O0000OOO0O000 .background_prior #line:42
    for OO000000O00OO0OO0 ,O0O0OO0OOOOO0OO00 in enumerate (OO00O0000OOO0O000 .site_base_probs ):#line:45
        O0O00OO0O00O0O00O *=O0O0OO0OOOOO0OO00 [O00OO00O0OO0OO00O [OO000000O00OO0OO0 ]]#line:46
        O0O0O0O0O0OOOO00O *=OO00O0000OOO0O000 .background_base_probs [O00OO00O0OO0OO00O [OO000000O00OO0OO0 ]]#line:48
    O0O0OOO00OOOO0OOO =(O0O00OO0O00O0O00O /(O0O00OO0O00O0O00O +O0O0O0O0O0OOOO00O ))#line:54
    return O0O0OOO00OOOO0OOO 