import sys
import argparse
import logging
import numpy as np
from goops import utils
from goops import motif

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Partitions sequences through  motifs discovery using the goops algorithm.")
    parser.add_argument("-o", "--output-prefix", type=str, default='goops_output', help='Prefix for file outputs. (Default: goops_output)')
    parser.add_argument("-m", "--min-length", type=int, default=8, help="Minimum motif length")
    parser.add_argument("-M", "--max-length", type=int, default=10, help="Maximum motif length")
    parser.add_argument("-a", "--algorithm", type=str, default="EM", help="Algorithm for motif discovery. Options are 'EM' (Expectation-Maximization) or 'MH' (Metropolis-Hastings)")
    parser.add_argument("--count", type=int, default=1, help="Number of times to repeat an action.")
    parser.add_argument("fasta", help="Input Fasta file.")
    args = parser.parse_args()

    # Check Algorithm Options
    if args.algorithm not in ["EM", "MH"]:
        parser.print_help()
        print("\nERROR: Algorithm must be either 'EM' or 'MH'.")
        sys.exit(1)

    # Read Fasta
    fasta = utils.read_fasta(args.fasta)

    # Create Goops object
    goops = motif.Goops(fasta)

    # Discover Motifs
    goops.discover(min_length = args.min_length,
                   max_length = args.max_length, 
                   algo = args.algorithm,
                   prefix = args.output_prefix)

if __name__ == "__main__":
    main()
