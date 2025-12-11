import sys
import math
import logging
import numpy as np
from goops import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


#########################################################################################################
# Motif Discovery Class
class Goops:

    ######################################################################
    # Constructor
    def __init__(self, sequeces: dict):
        
        # Input Data
        self.sequeces = sequeces
        self.num_seqs = len(sequeces)
        
        # Model Parameters
        self.num_groups = None
        self.min_length = None
        self.max_length = None
        self.num_lengths = None

        # Algorithm Parameters
        self.num_repeats = 1# 5
        self.min_itererations = 5
        self.max_itererations = None
        self.pseudocount = 0.001
        self.epsilon = 0.01
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
        self.max_itererations = max_iter



    ######################################################################
    # Evaluate Motif Likelihood
    def motif_ll(self, seq: str, _motif: np.ndarray, start_pos: int = None):
        
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
                    ll += math.log(0.25)

        else:
            for z in range(num_pos):
                end_pos = z + length - 1 
                for pos in range(n):    
                    b = self.bases[seq[pos]]
                    if pos >= z and pos <= end_pos:
                        ll += np.log(_motif[b][pos - z])
                    else:
                        ll += math.log(0.25)

        return ll


    ######################################################################
    # Initialize Motif Models
    def __initialize_models(self, uniform: bool = False):
        
        g_prob = 1 / self.num_groups
        l_prob = 1 / self.num_lengths
        n_prob = 0.25

        # Uniform Priors
        _gamma  = np.full((self.num_groups, self.num_seqs), g_prob, dtype=np.float64)
        _lambda = np.full((self.num_groups, self.num_lengths), l_prob, dtype=np.float64)
        _motifs = [[] for g in range(self.num_groups)]

        # Initlize Motif Model for all Groups and Lengths
        for g in range(self.num_groups):
            if not uniform:
                _motifs[g] = [utils.random_mat(4, l) for l in range(self.min_length, self.max_length+1)]
            else:
                _motifs[g] = [np.full((4, l), n_prob, dtype=np.float64) for l in range(self.min_length, self.max_length+1)]

        return _motifs, _lambda, _gamma


    ######################################################################
    # Expectation Maximization Goops Implementation
    def __discover_EM(self, _gamma: np.ndarray, _lambda: np.ndarray, _motifs: list):

        G = self.num_groups
        L = self.num_lengths
        groups = [g for g in range(G)]
        lengths = [l for l in range(self.min_length, self.max_length+1)]

        iterations = 0
        converged = False
        max_likelihood_length = np.zeros((G, L))
        while not converged and iterations < 20:

            log.info("Iteration: " + str(iterations))
            print(np.sum(_gamma, axis = 1))

            x = 0
            _motifs_tp1, _lambda_tp1, _gamma_tp1 = self.__initialize_models(uniform = True)            
            
            for header, seq in self.sequeces.items():
                
                for l in range(L):
                    
                    num_pos = len(seq) - lengths[l] + 1
                    Q = np.full((G, num_pos), math.log(self.pseudocount), dtype=np.float64)

                    # E Step          
                    for g in groups:                  
                        for z in range(num_pos):     
                            Q[g][z] += self.motif_ll(seq, _motifs[g][l], start_pos = z)
                            #Q[g][z] += np.log(_gamma[g][x]) + np.log(_lambda[g][l]) + np.log(1 / num_pos)
                            Q[g][z] += np.log(0.5) + np.log(_lambda[g][l]) + np.log(1 / num_pos)
                        Q[g,:] = utils.logsafe_normalize(Q[g,:])

                    Q_groups = Q.copy()
                    for pos in range(Q_groups.shape[1]):
                        Q_groups[:,pos] =  utils.logsafe_normalize(Q_groups[:,pos])

                    # M Step
                    """
                    Derivation of M step is not correct
                    we probably need to normalize across groups
                    for Q. 

                    """
                    for g in groups:
                        _gamma_tp1[g][x] += np.sum(np.exp(Q_groups[g,:]))
                        _lambda_tp1[g][l] += np.sum(np.exp(Q[g,:]))
                        for z in range(num_pos):
                            for m in range(lengths[l]):
                                for n, b in self.bases.items():
                                    if seq[z+m] == n:
                                        _motifs_tp1[g][l][b][m] += np.exp(Q[g][z])
                    # for g in groups:
                    #     for z in range(num_pos):
                    #         _gamma_tp1[g][x] += np.exp(Q[g][z])
                    #         _lambda_tp1[g][l] += np.exp(Q[g][z])
                    #         for m in range(lengths[l]):
                    #             for n, b in self.bases.items():
                    #                 if seq[z+m] == n:
                    #                     _motifs_tp1[g][l][b][m] += np.exp(Q[g][z])

                x += 1 # Incr seq count


            # Noramlize models to sum to 1
            _gamma_tp1 /= np.sum(_gamma_tp1, axis = 0)
            for g in groups:
                _lambda_tp1[g] /= np.sum(_lambda_tp1[g])
                for m in range(len(_motifs_tp1[g])):
                    for c in range(_motifs_tp1[g][m].shape[1]):
                         _motifs_tp1[g][m][:,c] /= np.sum(_motifs_tp1[g][m][:,c])

            # Check Convergence
            diff = np.abs(np.array((_gamma_tp1 - _gamma)))
            if iterations > self.min_itererations and bool(np.all(diff < self.epsilon)):
                converged = True

            _gamma = _gamma_tp1
            _lambda = _lambda_tp1
            _motifs = _motifs_tp1
            iterations += 1


        """
        12/11/25 BNJ: 
            As of now, this is only going to return the motifs of the shortest lengths.
            I envision the selection of the most likely motif and the classifications 
            should also go here.
        """

        seq_ids = list(self.sequeces.keys())
        classifications = np.argmax(_gamma, axis=0)
        results = {"Groups": {seq: int(group) for seq, group in zip(seq_ids, classifications)},
                   "Motifs": {}}

        for g in groups:
            results["Motifs"]["Group_" + str(g)] = {"Motif": _motifs[g][0], "LogLikelihood": ";)"}

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
        for i in range(self.num_repeats):

            """
            12/11/25 BNJ: 
                I Imagine this is were we will implement better landscape exploration
            """
        
            _motifs, _lambda, _gamma = self.__initialize_models()

            # Run Aglorithm
            if algo == "EM":
                results = self.__discover_EM(_gamma, _lambda, _motifs)
            else:
                print("ERROR: Algorithm " + algo + " is not implemented yet.")
                sys.exit(1)

        # if true, store results in Goops object
        if store:
            self.results = results

        return results




