import os
from pathlib import PosixPath
import numpy as np
from typing import List, Tuple
from Bio import SeqIO
from .BaseTranslator import BaseTranslator


def read_in_fasta(fasta_path: Tuple[str, PosixPath],
                  include_reverse_complement: bool = False,
                  **kwargs) -> List[List[int]]:
    """Read in a fasta file and return a list of sequences

    :param fasta_path: path to a fasta file. Note that the .fai index file
        must exist in the same directory as the fasta file
    :type fasta_path: str, PosixPath
    :param include_reverse_complement: Controls whether to include the reverse
        complements of the fasta sequences in the return. If True, then the
         input sequence will be in even indicies, plus zero, and the reverse
          complement will be in odd indicies. Defaults to False
    :type include_reverse_complement: bool, optional
    :raises FileNotFoundError: if the input fasta file does not exist, or the
        .fai index file does not exist in the same directory as the fasta
    :return: a numpy array of sequences translated from their str
        representations to numeric representations where 0=A, 1=C, 2=G, 3=T
    :rtype: 
    """
    if isinstance(fasta_path, PosixPath):
        fasta_path_string: str = str(fasta_path)
    else:
        fasta_path_string = fasta_path
    if not os.path.exists(fasta_path_string):
        raise FileNotFoundError(f"{fasta_path_string} does not exist.")
    if not os.path.exists(fasta_path_string + ".fai"):
        raise FileNotFoundError(f"{fasta_path} does not have a .fai index "
                                f"file. Create one with "
                                f"samtools faidx {fasta_path}")
    bt = BaseTranslator()
    num_records = len(SeqIO.index(str(fasta_path), "fasta"))
    # if include_reverse_complement is True, then the number of
    # sequence records is double
    if include_reverse_complement:
        sequences: List[List[int]] = [[0]] * (2 * num_records)
    else:
        sequences = [[0]] * num_records

    for i, record in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
        if include_reverse_complement:
            # input sequence are even indexed plus zero, 0, 2, 4, ...
            sequences[2*i] = bt.translate_char_to_int(str(record.seq))
            # Reverse complement of input are odd indexed, 1, 3, 5, ...
            reverse_complement_seq = record.seq.reverse_complement()
            sequences[2*i + 1] = bt.translate_char_to_int(
                str(reverse_complement_seq))
        else:
            sequences[i] = bt.translate_char_to_int(str(record.seq))

    return sequences