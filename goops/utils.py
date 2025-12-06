import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Read fasta utility
def read_fasta(filename: str):
	seqs = {}
	with open(filename, "r") as file:
		curr = ""
		for line in file:
			if line[0] == '>':
				curr = line.strip()
				seqs[curr] = ""
			else:
				seqs[curr] += line.strip()
	return seqs