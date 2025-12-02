import sys
import argparse
import logging
import numpy as np
from goops import utils
from goops import motif

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description="Partitions sequences through  motifs discovery using the goops algorithm.")
    parser.add_argument("-o", "--output-prefix", type=str, default='goops_output', help='Prefix for file outputs. (Default: goops_output)')
    parser.add_argument("fasta", help="Input Fasta file.")
    args = parser.parse_args()


    #fasta = util.readfasta(args.input)
    #output = motif.discover(fasta)
    # write output file i guess

if __name__ == "__main__":
    main()
