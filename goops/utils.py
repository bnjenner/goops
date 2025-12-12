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
def parse_arguments():

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Partitions sequences through  motifs discovery using the goops algorithm.")
    parser.add_argument("-o", "--output-prefix", type=str, default='goops_output', help='Prefix for file outputs. (Default: goops_output)')
    parser.add_argument("-g", "--groups", type=int, default=2, help="Number of motif groups (Min: 1)")
    parser.add_argument("-m", "--min-length", type=int, default=8, help="Minimum motif length")
    parser.add_argument("-M", "--max-length", type=int, default=10, help="Maximum motif length")
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
    # pwm = np.random.gamma(0.1, 1.0, size=(row, col))
    pwm = np.random.beta(100, 100, size=(row, col))
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
        pwm.T.reshape(pwm.shape[1], pwm.shape[0]),
        columns=["A", "C", "G", "T"],
        index=range(pwm.shape[1]),
    )
    return pwm


######################################################################
# Shannon's Entropy
def entropy(probs: np.array):
    nonzero = probs > 0
    return -np.sum(probs[nonzero] * np.log2(probs[nonzero]))


######################################################################
# Make Motif Logo
def make_logo(pwm: ndarray, prefix: str, scale_by_info: bool = True, write_pwm_tsv: bool = True):

    df = convert_to_df(pwm)

    # Write to TSV
    if write_pwm_tsv:
        df.to_csv(prefix + ".txt", index=False, sep="\t")


    # Scale by information content
    if scale_by_info:
        entropies = np.array([entropy(row) for row in pwm.T])
        df = df * (2 - entropies[:, None])

    # Generate Logo
    # logo = logomaker.Logo(df, shade_below=.5, fade_below=.5)
    logo = logomaker.Logo(df)
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)
    logo.ax.set_ylabel("Position")
    logo.ax.set_ylabel("Frequency")
    logo.ax.set_title(prefix.split("/")[-1])
    plt.savefig(prefix + ".png")
