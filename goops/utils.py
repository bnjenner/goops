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
# Random Biased PWM
def opp_mat(mat: np.ndarray):
	opp = 1.0 / mat
	opp /= opp.sum(axis=0, keepdims=True)
	return opp


def random_vec(length: int):
    pwm = np.random.beta(100, 100, size=length)
    pwm /= pwm.sum(axis=0, keepdims=True)
    return pwm


def make_logo(pwm, prefix: str):

    pwm = pd.DataFrame(
        pwm.T.reshape(pwm.shape[2], pwm.shape[1]),
        columns=["A", "C", "G", "T"],
        index=range(pwm.shape[2]),
    )


    # H = -(pwm * np.log2(pwm + 1e-9)).sum(axis=0)  # entropy per position
    # print(H)
    # R = 2 - H                                   # information content (DNA)
    # print(R)
    # pwm = pwm * H  # multiply each column by its information content


    logo = logomaker.Logo(pwm,
    					  shade_below=.5,
                          fade_below=.5)
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)
    logo.ax.set_ylabel("Position")
    logo.ax.set_ylabel("Frequency")
    logo.ax.set_title(prefix.split("/")[-1])
    plt.savefig(prefix + ".png")
