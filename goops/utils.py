import logging
import numpy as np
from scipy.special import logsumexp


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

######################################################################
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

######################################################################
# Buffer underflow safe log sum
def logsafe_normalize(logs: np.ndarray):
	print(logs)
	log_sum = logsumexp(logs) 
	logP = logs - log_sum
	print(logP)
	return logP