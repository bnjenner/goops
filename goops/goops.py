import sys
import argparse
import logging
import numpy as np
from goops import utils
from goops import motif

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


#########################################################################################################
# Parse Arguments
def parse_arguments():

    # Usage Message
    parser = argparse.ArgumentParser(description="Partitions sequences through  motifs discovery using the goops algorithm.")
    parser.add_argument("-o", "--output-prefix", type=str, default='goops_output', help='Prefix for file outputs. (Default: goops_output)')
    parser.add_argument("-g", "--groups", type=int, default=2, help="Number of motif groups (Min: 1)")
    parser.add_argument("-m", "--min-length", type=int, default=8, help="Minimum motif length. If min = max lengths, only that length will be considered.")
    parser.add_argument("-M", "--max-length", type=int, default=10, help="Maximum motif length. If min = max lengths, only that length will be considered.")
    parser.add_argument("-a", "--algorithm", type=str, default="EM", help="Algorithm for motif discovery. Options are 'EM' (Expectation-Maximization) or 'MH' (Metropolis-Hastings)")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="Maximum number of iterations.")
    parser.add_argument("fasta", help="Input Fasta file.")
    args = parser.parse_args()

    # Check Args Options
    if args.algorithm not in ["EM", "MH"]:
        parser.print_help()
        print("\nERROR: Algorithm must be either 'EM' or 'MH'.")
        sys.exit(1)
    if args.groups < 1:
        parser.print_help()
        print("\nERROR: Group Number must be greater than or equal to 1.")
        sys.exit(1)
    if args.iterations < 1:
        parser.print_help()
        print("\nERROR: Max Iterations must be greater than or equal to 1.")
        sys.exit(1)
    if args.min_length > args.max_length:
        parser.print_help()
        print("\nERROR: Min Length greater than Max Length.")
        sys.exit(1)

    args = parser.parse_args()
    return args


#########################################################################################################
# Main
def main():

    # Parse Arguments
    args = parse_arguments()

    # Read Fasta
    fasta = utils.read_fasta(args.fasta)

    # Create Goops object
    goops = motif.Goops(fasta)

    # Set Goops parameters
    goops.set_parameters(num_groups = args.groups,
                         min_length = args.min_length,
                         max_length = args.max_length,
                         max_iter = args.iterations)

    # Discover Motifs
    results = goops.discover(algo = args.algorithm)

    # Write output
    if  results is not None:
        for group, res in results["Motifs"].items():
            utils.make_logo(res["Motif"], args.output_prefix + "_" + group)

    print(results["Groups"], file=sys.stderr)
    for group, res in results["Motifs"].items():
        print(group, file=sys.stderr)
        print(res, file=sys.stderr)

if __name__ == "__main__":
    main()
