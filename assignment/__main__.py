import argparse
import sys
import os
import numpy as np
from .assignment import siteEM_intializer
from .utils.consensus_pfm import consensus_pfm


def parse_args(args: list) -> argparse.Namespace:
    """
    Parse command line arguments

    :param args: command line arguments
    :type args: list

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    description = "Run siteEM on a fasta file"
    parser = argparse.ArgumentParser(description=description)

    # Input Options
    input_options_parser = parser.add_argument_group('Input Options')
    input_options_parser.add_argument("-f", "--fasta",
                                      dest="fasta",
                                      type=str,
                                      help="fasta file to run siteEM on")
    input_options_parser.add_argument("-m", "--motif_length",
                                      dest="motif_length",
                                      type=int,
                                      help="length of motif to search for")
    input_options_parser.add_argument("-n", "--num_motifs_to_find",
                                      dest="num_motifs_to_find",
                                      type=int,
                                      default=1,
                                      help="number of sites to search for. "
                                      "Defaults to 1")
    input_options_parser.add_argument("-p", "--pseudocount",
                                      default=0.01,
                                      type=float,
                                      dest="pseudocount",
                                      help="pseudocount to use in siteEM. "
                                      "Defaults to 0.01")
    input_options_parser.add_argument("-i", "--max_iterations",
                                      dest="max_iterations",
                                      default=1000,
                                      type=int,
                                      help="Maximum number of iterations "
                                      "which the EM algorithm may run. "
                                      "Defaults to 1000")
    input_options_parser.add_argument('-a', '--accuracy',
                                      default=1e-6,
                                      dest='accuracy',
                                      type=float,
                                      help='accuracy stopping condition of '
                                      'siteEM. Defaults to 1e-6')
    input_options_parser.add_argument('-r', '--include_reverse_complement',
                                      default=False,
                                      dest='include_reverse_complement',
                                      action='store_true',
                                      help='include `-r` flag to include the '
                                      'reverse complement of the input '
                                      'sequences.')
    input_options_parser.add_argument('-s', '--set_site_probs_seed',
                                      default=False,
                                      dest='site_base_probs_seed',
                                      action='store_true',
                                      help='Set the `-s` flag to set the site '
                                      'the site base probs seed to 42 for '
                                      'consistency. If this flag is not '
                                      'set, then the seed is set randomly.')

    # Output Options
    output_options_parser = parser.add_argument_group('Output Options')
    output_options_parser.add_argument("-o", "--output",
                                       dest="output",
                                       help="output file to write results to")

    return parser.parse_args(args)


def main(args: list) -> None:
    """
    Execute siteEM on a fasta file. Save results to file.

    :param args: command line arguments
    :type args: list

    :return: None
    :rtype: None

    :raises FileNotFoundError: if fasta file or fasta index file not found
    :raises FileExistsError: if output directory does not exist

    :Example:

    $ poetry run python -m assignment -f assignment/data/PACPlusSeqs.fasta -m 8 -a 0.01 -i 100 -n 3 -o pac_test.txt -r -s
    """

    args = parse_args(args)

    if not os.path.exists(args.fasta):
        raise FileNotFoundError("fasta file not found")
    if not os.path.exists(args.fasta+'.fai'):
        raise FileNotFoundError("fasta index file not found. Create the "
                                "index with samtools faidx")
    if not os.path.exists(os.path.dirname(args.output) or os.getcwd()):
        raise FileExistsError("output directory does not exist")

    # run siteEM
    results = siteEM_intializer(
        args.fasta,
        args.motif_length,
        args.pseudocount,
        accuracy=args.accuracy,
        max_iterations=args.max_iterations,
        num_motifs_to_find=args.num_motifs_to_find,
        site_base_probs_seed=42 if args.site_base_probs_seed else None,
        include_reverse_complement=args.include_reverse_complement)

    # write results to file
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, motif_res in enumerate(results):
            f.write('motif: '
                    + str(i)
                    + '\n\t')
            f.write('consensus: '
                    + consensus_pfm(motif_res[1].site_base_probs)
                    + '\n\t')
            f.write('pfm: ' + str(np.array(motif_res[1].site_base_probs))
                    + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
