# pylint: disable=C0103
from typing import List, Union, Tuple, Optional
import logging
from math import inf
from copy import deepcopy
from pathlib import PosixPath
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from .utils.bailey_elkan_hacks import \
    normalize_posterior_row, update_eraser_row
from .utils.read_in_fasta import read_in_fasta
from .utils.nucleotide_count import nucleotide_count

def seq2onehot(sequence:List[int],num_classes:int) -> NDArray:
    seq_len = len(sequence)
    onehot = np.eye(num_classes)[sequence]
    return onehot

def standardize_sequences(sequences):
    if not isinstance(sequences[0], np.ndarray):
        onehot_sequences = [seq2onehot(seq,4) for seq in sequences]
    else:
        onehot_sequences = sequences
    return onehot_sequences

# def batch_seq2onehot(sequences: List[List[int]],num_classes:int) -> LiNDArray:
    # # sequence_onehot = np.concatenate([
        # # # [N,L] -> [N,L,b]
        # # # N: # of seq 
        # # # L: len of seq 
        # # # b: # of bases i.e. 4
        # # np.expand_dims(seq2onehot(seq, num_classes=4),0) 
        # # for seq in sequences
    # # ], axis=0)
    # return sequence_onehot

def onehot_seq2windows(onehot_seq:NDArray, window_size:int, batch=False) -> NDArray:
    # onehot_seq: [L, b] where b =4
    if batch:
        raise NotImplementedError
    return sliding_window_view(onehot_seq, (window_size, onehot_seq.shape[1]))[:,0]


class LikelihoodSequenceModel(SequenceModel):
    def sample_motif(self):
        vectorized_choice = np.vectorize(
            # 4 base 
            pyfunc=lambda prob_base_i: np.random.choice(4, size=1, p=prob_base_i),
            signature="(m) -> ()"
        )
        return vectorized_choice(self.site_base_probs)

    def sample_background(self):
        # print("size",self.motif_length, "p",self.background_base_probs)
        return np.random.choice(
            4, # # mapping: 0 = A, 1 = C, 2 = G, 3 = T
            size= self.motif_length(),
            replace=True, 
            p=self.background_base_probs 
        )

    def likelihood_motif(self, sequence_onehot, batched=False):
        # sequence_onehot: [seq_len, 4]
        # base_prob: [seq_len, 4]
        # likelihood: prod_i sum_k(p_base_ik * delta_ik)
        if batched:
            return np.prod(np.einsum(
                "nlb, lb->nl",
                sequence_onehot, self.site_base_probs
            ), axis=1)
        else:
            return np.prod(np.einsum(
                "lb, lb->l",
                sequence_onehot, self.site_base_probs
            ))

    def likelihood_background(self, sequence_onehot, batched=False):
        # sequence_onehot: [seq_len, 4]
        # base_prob: [4 ]
        if batched:
            return np.prod(np.einsum(
                "nlb, b->nl",
                sequence_onehot, self.background_base_probs
            ), axis=1)
        else:
            return np.prod(np.einsum(
                "lb, b->l",
                sequence_onehot, self.background_base_probs
            ))

    @classmethod
    def from_super(cls, obj:SequenceModel) -> "LikelihoodSequenceModel":
        return LikelihoodSequenceModel(
                obj.site_prior, obj.site_base_probs, obj.background_base_probs,
                obj.precision, obj.tolerance)

logger = logging.getLogger(__name__)


# e_step takes a single input sequence, not the list of all input sequences.
def e_step(sequence: Union[List[int], NDArray],
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
    if isinstance(sequence, list): # convert sequence to onehot np array [L, b]
        sequence_onehot = seq2onehot(sequence, 4)
    else:
        sequence_onehot = sequence  
    if not isinstance(sequence_model, LikelihoodSequenceModel):
        sequence_model = LikelihoodSequenceModel.from_super(sequence_model)
    if not sequence_onehot.shape[0] >= len(sequence_model):
        raise ValueError("sequence must be longer than site_base_probs")
    if not isinstance(sequence_model, SequenceModel):
        raise TypeError("sequence_model must be a SequenceModel object")
    
    # posteriors_row is the NDArray of posteriors for a single input sequence.
    # sequence_onehot: [L,b]
    # convert to slide window [L-W+1, W, b]
    windows_onehot = onehot_seq2windows(sequence_onehot, len(sequence_model)) 
    posteriors_row =  site_posterior(
        batched_window_onehot=windows_onehot,
        sequence_model=sequence_model
    )
    # iterate a sliding window of length motif_length over the sequence
    # eg. for motif length 2 and sequence [0, 1, 2, 3], the
    # windows are [0, 1], [1, 2], [2, 3]. Your code will call site_posterior on 
    # each of these windows and store the result in posteriors_row.
    # remember that python is 0 indexed and exclusive of the end index
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
    posteriors_all_motif = np.concatenate(posteriors, axis=0)
    motif_marginal = np.mean(posteriors_all_motif)
    return motif_marginal


def update_site_probs(sequences: Union[List[List[int]], List[NDArray]],
                      motif_length: int,
                      posteriors: List[NDArray[np.float64]],
                      motif_pseudocounts: Union[List[float],NDArray],
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
    :type motif_pseudocounts: list[float]
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
    # convert sequences to onehot 
    onehot_sequences = standardize_sequences(sequences)
    n_seq = len(onehot_sequences)
    assert n_seq == len(posteriors), """
    number of sequences should equal to number of posteriors.
    got {} and {}
    """.format(n_seq, len(posteriors))
    if isinstance(motif_pseudocounts, list):
        motif_pseudocounts = np.array(motif_pseudocounts) # [4]
    if erasers:
        onehot_sequences = [
            np.einsum(
                "lb,l->lb", # set one hot indicator to 0 if position is erased
                onehot_sequences[i_seq], erasers[i_seq]
            ) for i_seq in range(n_seq)
        ]
    onehot_windows = [
        onehot_seq2windows(seq, motif_length)
        for seq in onehot_sequences
    ]
    # instantiate a list of lists to hold the unnormalized site probs.
    # unnormalized_site_probs = [deepcopy(motif_pseudocounts)
                               # for _ in range(motif_length)]
    # site probs: [W,b]
    # freqs: [W,b]
    # update the pfm by looping over sequences, windows within each sequence,
    # and positions within each window. When calculating the expected frequency
    # of a given letter in a given position of the motif, multiply posteriors by 
    # erasers. Note the lenghth of the erasers is that of the sequence, not that
    # of the posteriors -- you'll need to index into the erasers list to get the
    # correct eraser for the current position in the motif. You will need 3 nested
    # loops to do this.
    # calculate the sum of each position in the pfm
    expected_base_freq_in_motif = sum([
        # for each sequence 
        np.einsum(
            "nwb, n -> wb",
            onehot_windows[i_seq], posteriors[i_seq]
        ) for i_seq in range(n_seq)
    ])
    # add by pseudocount
    unnormalized_site_probs = expected_base_freq_in_motif + motif_pseudocounts[np.newaxis,:]
    normalizer = np.sum(unnormalized_site_probs, axis=1)[:, np.newaxis]

    return (unnormalized_site_probs / normalizer) if normalize \
        else unnormalized_site_probs


def m_step(sequences: Union[List[List[int]], List[NDArray]],
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
    onehot_sequences = standardize_sequences(sequences)
    # def update_motif_prior(posteriors: List[NDArray[np.float64]]) -> float:
    updated_site_prior = update_motif_prior(posteriors)
    updated_site_probs = update_site_probs(onehot_sequences,motif_length,posteriors,motif_pseudocounts,erasers)
    # Update the motif prior and site probabilities based on the posteriors.
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
    if not isinstance(init_sequence_model, LikelihoodSequenceModel):
        init_sequence_model = LikelihoodSequenceModel.from_super(init_sequence_model)
    motif_length = len(init_sequence_model)
    onehot_sequences = [
        seq2onehot(seq, 4) for seq in sequences
    ]
    
    # Perform EM iterations
    for index in range(num_motifs_to_find):
        iteration = 0
        parameter_diff = inf
        current_sequence_model = init_sequence_model
        while parameter_diff > accuracy and iteration < max_iterations:
            # E-step
            # Note: the E-step is run on each sequence in the input list of
            normalized_posteriors: List[NDArray] = [
                e_step(seq, current_sequence_model)
                for seq in onehot_sequences
            ]
            # sequences
            # M-step. The M-step requires the entire list of sequences.
            updated_site_prior, updated_site_probs = m_step(
                sequences,motif_length,normalized_posteriors,motif_pseudocounts,erasers
            )
            # set the prev_sequence_model to a copy of the parameters which
            # where used in this iteration
            prev_sequence_model = current_sequence_model
            # store the updated parameters in a SequenceModel object.
            # the background probs do not change between iterations
            current_sequence_model = LikelihoodSequenceModel(
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



def site_posterior(
        batched_window_onehot: NDArray,
        sequence_model: LikelihoodSequenceModel) -> NDArray:
    """
    Calculate the posterior probability of a bound site versus an unbound site.

    :param batched_window_onehot [N,W,b]  
    :return: Posterior probability of a bound site [N]
    """
    # check that the inputs are valid
    numerator_site = sequence_model.site_prior * sequence_model.likelihood_motif(batched_window_onehot, batched=True)
    numerator_bg = (1-sequence_model.site_prior) * sequence_model.likelihood_background(batched_window_onehot, batched=True)
    if np.any((numerator_site == 0) * (numerator_bg == 0)):
        raise ZeroDivisionError("got likelihood be 0 for both site and background")
    posterior_prob = numerator_site / (numerator_site + numerator_bg)
    return posterior_prob

# def site_posterior (O00OO00O0OO0OO00O :list [int ],OO00O0000OOO0O000 :SequenceModel )->float :#line:2
    # ""#line:25
    # if not isinstance (O00OO00O0OO0OO00O ,list ):#line:27
        # raise TypeError ("sequence must be a list")#line:28
    # if not len (O00OO00O0OO0OO00O )==OO00O0000OOO0O000 .motif_length ():#line:29
        # raise ValueError ("sequence and site_base_probs must be the same length")#line:31
    # for O0OOOOOOO0O000OOO in O00OO00O0OO0OO00O :#line:32
        # if not isinstance (O0OOOOOOO0O000OOO ,int ):#line:33
            # raise TypeError ("sequence must be a list of integers")#line:34
        # if O0OOOOOOO0O000OOO <0 or O0OOOOOOO0O000OOO >3 :#line:35
            # raise ValueError ("sequence must be a list of integers between 0 " "and 3 (inclusive)")#line:37
    # O0O00OO0O00O0O00O =OO00O0000OOO0O000 .site_prior #line:41
    # O0O0O0O0O0OOOO00O =OO00O0000OOO0O000 .background_prior #line:42
    # for OO000000O00OO0OO0 ,O0O0OO0OOOOO0OO00 in enumerate (OO00O0000OOO0O000 .site_base_probs ):#line:45
        # O0O00OO0O00O0O00O *=O0O0OO0OOOOO0OO00 [O00OO00O0OO0OO00O [OO000000O00OO0OO0 ]]#line:46
        # O0O0O0O0O0OOOO00O *=OO00O0000OOO0O000 .background_base_probs [O00OO00O0OO0OO00O [OO000000O00OO0OO0 ]]#line:48
    # O0O0OOO00OOOO0OOO =(O0O00OO0O00O0O00O /(O0O00OO0O00O0O00O +O0O0O0O0O0OOOO00O ))#line:54
    # return O0O0OOO00OOOO0OOO 
