import logging
import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    pwm = np.random.gamma(0.1, 1.0, size=(row, col))
    # pwm = np.random.beta(100, 100, size=(row, col))
    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm

######################################################################
# Random Opposite PWM
def opp_mat(mat: np.ndarray):
	opp = 1.0 / mat
	opp /= opp.sum(axis=0, keepdims=True)
	return opp


######################################################################
# Random Biased Vector
def random_vec(length: int):
    pwm = np.random.beta(100, 100, size=length)
    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm


######################################################################
# Convert to Pandas DF
def convert_to_df(pwm: ndarray):
    pwm = pd.DataFrame(
        pwm.T.reshape(pwm.shape[2], pwm.shape[1]),
        columns=["A", "C", "G", "T"],
        index=range(pwm.shape[2]),
    )
    return pwm


######################################################################
# Shannon's Entropy
def entropy(probs: np.array):
    nonzero = probs > 0
    return -np.sum(probs[nonzero] * np.log2(probs[nonzero]))


######################################################################
# Make Motif Logo
def make_logo(pwm: ndarray, prefix: str):

    entropies = np.array([entropy(row) for row in pwm.T])
    pwm = convert_to_df(pwm)
    pwm = pwm * (2 - entropies[:, None]) # Scale by information content

    logo = logomaker.Logo(pwm,
    					  shade_below=.5,
                          fade_below=.5)
    logo = logomaker.Logo(pwm)
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)
    logo.ax.set_ylabel("Position")
    logo.ax.set_ylabel("Frequency")
    logo.ax.set_title(prefix.split("/")[-1])
    plt.savefig(prefix + ".png")
