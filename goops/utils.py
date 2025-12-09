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
	temp_log = logs
	log_sum = logsumexp(temp_log) 
	logP = temp_log - log_sum
	return logP


######################################################################
# Random Biased PWM
def random_mat(row: int, col: int):
    # pwm = np.random.rand(row, col)
    pwm = np.random.gamma(0.25, 1.0, size=(row, col))
    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm


def random_vec(length: int):
    pwm = np.random.gamma(0.5, 1.0, size=length)
    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm