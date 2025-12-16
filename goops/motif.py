import sys
import math
import logging
import numpy as np
from goops import utils
from scipy.special import kl_div

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


#########################################################################################################
# Motif Discovery Class
class Goops:

    ######################################################################
    # Constructor
    def __init__(self, sequences: dict):
        
        # Input Data
        self.sequences = sequences
        self.num_seqs = len(sequences)
        
        # Model Parameters
        self.num_groups = None
        self.min_length = None
        self.max_length = None
        self.num_lengths = None

        # Algorithm Parameters
        self.num_repeats = 5 # number of iterations to explore start points
        self.explore_num = 3 # number of start points for likelihood landscape exploration. Make this <1 to do likelihood landscape exploration.
        self.min_iterations = 5
        self.max_iterations = None
        self.pseudocount = 0.001
        self.epsilon = 0.0001
        self.bases = {"A": 0, "C": 1, "G": 2, "T": 3}

        # Results
        self.results = None


    ######################################################################
    # Set Clustering Parameters
    def set_parameters(self, num_groups: int, min_length: int, max_length: int, max_iter: int):
        self.num_groups = num_groups        
        self.min_length = min_length
        self.max_length = max_length
        self.num_lengths = max_length - min_length + 1
        self.max_iterations = max_iter



    ######################################################################
    # Evaluate Motif Likelihood
    def motif_ll(self, seq: str, _motif: np.ndarray, _background: np.ndarray,start_pos: int = None):
        
        ll = 0
        n = len(seq)
        length = _motif.shape[1]
        num_pos = n - length + 1

        if start_pos is not None:
            z = start_pos
            end_pos = z + length - 1 
            for pos in range(n):    
                b = self.bases[seq[pos]]
                if pos >= z and pos <= end_pos:
                    ll += np.log(_motif[b][pos - z])
                else:
                    ll += np.log(_background[b])

        else:
            for z in range(num_pos):
                end_pos = z + length - 1 
                for pos in range(n):    
                    b = self.bases[seq[pos]]
                    if pos >= z and pos <= end_pos:
                        ll += np.log(_motif[b][pos - z])
                    else:
                        ll += np.log(_background[b])

        return ll


    ######################################################################
    # Classify Sequences
    
    def classify_seq(self, seq_dict: dict):

        if self.results is None:
            print("ERROR: Motif discovery not performed yet.")
            sys.exit(1)

        n = len(self.results["Motifs"])
        classifications = {}
        for header, seq in seq_dict.items():

            ll = np.zeros(n, dtype=np.float64)
            classifications[header] = {"ID": header, "Class": None}
            
            i = 0
            for g, m in self.results["Motifs"].items():
                ll[i] = self.motif_ll(seq, m["Motif"], self.results["Background"])
                classifications[header]["LL_Group_" + str(i)] = str(round(ll[i], ndigits = 2))
                i += 1
            
            classifications[header]["Class"] = str(np.argmax(ll))

        return classifications


    ######################################################################
    # Initialize Motif Models
    def __initialize_models(self, uniform: bool = False, zeroes: bool = False, a: int = 100):
        
        if zeroes == True:
            g_prob = 0
            l_prob = 0
            n_prob = 0
        else:
            g_prob = 1 / self.num_groups
            l_prob = 1 / self.num_lengths
            n_prob = 0.25

        # Uniform Priors
        _gamma  = np.full((self.num_groups, self.num_seqs), g_prob, dtype=np.float64)
        _lambda = np.full((self.num_groups, self.num_lengths), l_prob, dtype=np.float64)
        _motifs = [[] for g in range(self.num_groups)]
        _background = [0.25, 0.25, 0.25, 0.25]

        # Initlize Motif Model for all Groups and Lengths
        for g in range(self.num_groups):
            if not uniform:
                if not zeroes:
                    _motifs[g] = [utils.random_mat(a, 4, l) for l in range(self.min_length, self.max_length+1)]
                else:
                    _motifs[g] = [np.full((4, l), 0, dtype = np.float64) for l in range(self.min_length, self.max_length+1)]
            else:
                _motifs[g] = [np.full((4, l), n_prob, dtype=np.float64) for l in range(self.min_length, self.max_length+1)]

        return _motifs, _lambda, _gamma, _background


    ######################################################################
    # Expectation Maximization Goops Implementation
    def __discover_EM(self, _gamma: np.ndarray, _lambda: np.ndarray, _motifs: list, _background: list, explore: bool = False):

        G = self.num_groups
        L = self.num_lengths
        groups = [g for g in range(G)]
        lengths = [l for l in range(self.min_length, self.max_length+1)]
        if explore:
            max_iter = self.num_repeats
        else:
            max_iter = max(self.min_iterations, self.max_iterations) - self.explore_num

        iterations = 0
        converged = False
        max_likelihood_length = np.zeros((G, L))
        while not converged and iterations < max_iter:
            if not explore and self.explore_num > 1:
                log.info("Iteration: " + str(iterations + self.explore_num))
                print(np.sum(_gamma / self.num_seqs, axis = 1))
            elif not explore:
                log.info("Iteration: " + str(iterations))
                print(np.sum(_gamma / self.num_seqs, axis = 1))
            # cont = input("Continue...")

            x = 0
            #_motifs_tp1, _lambda_tp1, _gamma_tp1, _background_tp1 = self.__initialize_models(zeroes = True)
            _motifs_tp1, _lambda_tp1, _gamma_tp1, _background = self.__initialize_models(zeroes = True)

            for header, seq in self.sequences.items():

                # print(header, _gamma[:,x])
                
                for l in range(L):
                    
                    num_pos = len(seq) - lengths[l] + 1
                    Q = np.full((G, num_pos), math.log(self.pseudocount), dtype=np.float64)

                    # E Step 
                    for z in range(num_pos):         
                        for g in groups:     
                            Q[g][z] += self.motif_ll(seq, _motifs[g][l], _background, start_pos = z)
                            Q[g][z] += np.log(_gamma[g][x]) + np.log(_lambda[g][l]) + np.log(1 / num_pos)
                        Q[:,z] =  utils.logsafe_normalize(Q[:,z])
                        # Q[g,:] = utils.logsafe_normalize(Q[g,:])

                    # for pos in range(Q.shape[1]):
                    #     Q[:,pos] =  utils.logsafe_normalize(Q[:,pos])


                    # M Step
                    for g in groups:
                        _gamma_tp1[g][x] += np.sum(np.exp(Q[g,:]))
                        _lambda_tp1[g][l] += np.sum(np.exp(Q[g,:]))
                        for z in range(num_pos):
                            for m in range(lengths[l]):
                                for n, b in self.bases.items():
                                    if seq[z+m] == n:
                                        _motifs_tp1[g][l][b][m] += (np.exp(Q[g][z]))
                                        #_background_tp1[b] += (1 - np.exp(Q[g][z]))

                x += 1 # Incr seq count
            
            # Noramlize models to sum to 1
            _gamma_tp1 /= np.sum(_gamma_tp1, axis = 0)
            #_background_tp1 /= np.sum(_background_tp1)
            for g in groups:
                _lambda_tp1[g] /= np.sum(_lambda_tp1[g])
                for m in range(len(_motifs_tp1[g])):
                    for c in range(_motifs_tp1[g][m].shape[1]):
                        _motifs_tp1[g][m][:,c] /= np.sum(_motifs_tp1[g][m][:,c])
                        _IC = kl_div(_motifs_tp1[g][m][:,c], _background)
                        _motifs_tp1[g][m][:,c] *= _IC
                        _motifs_tp1[g][m][:,c] /= np.sum(_motifs_tp1[g][m][:,c])


            # Check Convergence
            diff = np.abs(np.subtract(np.array(_gamma_tp1), np.array(_gamma)))
            if iterations > self.min_iterations and bool(np.all(diff < self.epsilon)):
                converged = True

            _gamma = _gamma_tp1
            _lambda = _lambda_tp1
            _motifs = _motifs_tp1
            #_background = _background_tp1
            iterations += 1


        """
        12/11/25 BNJ: 
            As of now, this is only going to return the motifs of the shortest lengths.
            I envision the selection of the most likely motif and the classifications 
            should also go here.
        """

        final_lens = np.argmax(_lambda, axis = 1)
        seq_ids = list(self.sequences.keys())
        classifications = np.argmax(_gamma, axis=0)
        results = {"Groups": {seq: int(group) for seq, group in zip(seq_ids, classifications)},
                   "Motifs": {}, "Background": _background}
        
        for g in groups:
            results["Motifs"]["Group_" + str(g)] = {"Motif": _motifs[g][final_lens[g]], "LogLikelihood": ";)"}

        if explore:
            params = {"Motif": _motifs, "Gamma": _gamma, "Lambda": _lambda, "Background": _background}
            return results, params
        else:
            return results


    ######################################################################
    # Metropolis-Hastings Goops Implementation
    def __discover_MH(self, models: list):
        return

    ######################################################################
    # Discover Motif Auxillary Functions
    def discover(self, algo: str, store = False):

        """
        MODEL PARAMETERS:

            _gamma: Group parameter matrix.
                - Dim: G x Number of Sequences

            _lambda: Length parameter matrix.
                - Dim: G x Number of Motif Lengths

            _motifs: List (groups) or lists (lengths) of Motif model matrices
                - Length: G
                - Length: Number of Motif Lengths
                - Dim: 4 x Length of Motif

        """

        results = None
        if self.explore_num > 1:
            LL_dict = {}
            for i in range(self.explore_num):

                a = (1 + i) * (100/self.num_repeats)
            
                _motifs, _lambda, _gamma, _background = self.__initialize_models(a = a)

                # Run Aglorithm
                if algo == "EM":
                    log.info("Exploration " + str(i) + ", Exploring Likelihood Landscape")
                    results, params = self.__discover_EM(_gamma, _lambda, _motifs, _background, True)
                    self.results = results
                else:
                    print("ERROR: Algorithm " + algo + " is not implemented yet.")
                    sys.exit(1)
                classifications = self.classify_seq(self.sequences)
                self.results = None
                iter_LL = sum([float(classifications[header]["LL_Group_" + str(g)])/self.motif_ll(seq, np.array([[b] for b in params["Background"]]), params["Background"]) for header, seq in self.sequences.items() for g in range(self.num_groups)])
                LL_dict[iter_LL] = params
            best_iter = LL_dict[max(LL_dict.keys())]
            results = self.__discover_EM(best_iter["Gamma"], best_iter["Lambda"], best_iter["Motif"], best_iter["Background"], False)
        else:
            _motifs, _lambda, _gamma, _background = self.__initialize_models(a = 100)
            results = self.__discover_EM(_gamma, _lambda, _motifs, _background, False)

        # if true, store results in Goops object
        if store:
            self.results = results

        return results




