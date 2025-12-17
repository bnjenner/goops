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
        
        # Model Parameters (Defaults)
        self.num_groups = 8
        self.min_length = 8
        self.max_length = 12
        self.num_lengths = self.max_length - self.min_length + 1

        # Algorithm Parameters
        self.num_repeats = 5 # number of iterations to explore start points
        self.explore_num = 3 # number of start points for likelihood landscape exploration. Make this <1 to do likelihood landscape exploration.
        self.min_iterations = 5
        self.max_iterations = 20
        self.pseudocount = 0.001
        self.epsilon = 0.0001
        self.bases = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.kl_weights = True

        # Results
        self.results = None


    ######################################################################
    # Set Clustering Parameters
    def set_parameters(self, num_groups: int, min_length: int, max_length: int, max_iter: int, explore_num: int, disable_kl: bool):
        self.num_groups = num_groups        
        self.min_length = min_length
        self.max_length = max_length
        self.num_lengths = max_length - min_length + 1
        self.max_iterations = max_iter
        self.explore_num = explore_num
        self.kl_weights = not disable_kl


    ######################################################################
    # Evaluate Motif Likelihood
    def motif_ll(self, seq: str, _motif: np.ndarray, _background: np.ndarray, start_pos: int = None):
        
        ll = 0
        n = len(seq)
        length = _motif.shape[1]
        num_pos = n - length + 1

        if start_pos is not None:

            # One starting position
            z = start_pos
            end_pos = z + length - 1 
            for pos in range(n):    
                b = self.bases[seq[pos]]
                if pos >= z and pos <= end_pos:
                    ll += np.log(_motif[b][pos - z])
                else:
                    ll += np.log(_background[b])

        else:

            # Any possible starting position
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
    def classify_seq(self, seq_dict: dict, motifs: dict = None, background: np.ndarray = np.full(4, 0.25, dtype=np.float64)):

        '''
        Takes a Dictionary of motif models 

        motifs = {"Group1": <model_1>,
                  "Group2": <model_2>}

        '''

        # Error Handling
        if motifs is None and self.results is None:
            print("ERROR: Motif discovery not performed yet.")
            sys.exit(1)
        elif motifs is None and self.results is not None:
            motifs = self.results["Groups"]

        predict = {}
        n = len(motifs)   
        for header, seq in seq_dict.items():
            
            predict[header] = {"ID": header, "Class": None}
            ll = np.zeros(n, dtype=np.float64)
            
            for g, m in motifs.items():
                ll[g] = self.motif_ll(seq, m["Motif"], background)
                predict[header]["LL_Group_" + str(g)] = str(round(ll[g], ndigits = 2))
            
            predict[header]["Class"] = str(np.argmax(ll))

        return predict


    ######################################################################
    # Initialize Motif Models
    def __initialize_models(self, uniform: bool = False, zeroes: bool = False, a: int = 100):
        
        # Zeroes or 
        if zeroes:
            g_prob = self.pseudocount 
            l_prob = self.pseudocount 
            n_prob = self.pseudocount 
        else:
            g_prob = 1 / self.num_groups
            l_prob = 1 / self.num_lengths
            n_prob = 0.25

        # Uniform Priors
        _gamma = np.full((self.num_groups, self.num_seqs), g_prob, dtype=np.float64)
        _lambda = np.full((self.num_groups, self.num_lengths), l_prob, dtype=np.float64)
        _background = np.full(4, 0.25, dtype=np.float64)
        _motifs = [[] for g in range(self.num_groups)]

        # Initlize Motif Model for all Groups and Lengths
        for g in range(self.num_groups):
            if not uniform:
                _motifs[g] = [utils.random_mat(a, 4, l) for l in range(self.min_length, self.max_length+1)]
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
        
        # Max Iterations based on exploration
        if explore:
            max_iter = self.num_repeats
        else:
            max_iter = self.max_iterations - self.explore_num

        # Main GOOPS EM Loop
        iterations = 0
        converged = False
        while not converged and iterations < max_iter:
           

            # if not explore and self.explore_num > 1:
            #     log.info("Iteration: " + str(iterations + self.explore_num))
            #     print(np.sum(_gamma / self.num_seqs, axis = 1))
            # elif not explore:
            #     log.info("Iteration: " + str(iterations))
            #     print(np.sum(_gamma / self.num_seqs, axis = 1))

            x = 0

            _motifs_tp1, _lambda_tp1, _gamma_tp1, _background = self.__initialize_models(zeroes = True)

            for header, seq in self.sequences.items():
                
                for l in range(L):
                    
                    num_pos = len(seq) - lengths[l] + 1
                    Q = np.full((G, num_pos), math.log(self.pseudocount), dtype=np.float64)

                    # E Step 
                    for z in range(num_pos):         
                        for g in groups:     
                            Q[g][z] += self.motif_ll(seq, _motifs[g][l], _background, start_pos = z)
                            Q[g][z] += np.log(_gamma[g][x]) + np.log(_lambda[g][l]) + np.log(1 / num_pos)
                        Q[:,z] =  utils.logsafe_normalize(Q[:,z])


                    # M Step
                    for g in groups:
                        _gamma_tp1[g][x] += np.sum(np.exp(Q[g,:]))
                        _lambda_tp1[g][l] += np.sum(np.exp(Q[g,:]))
                        for z in range(num_pos):
                            for m in range(lengths[l]):
                                for n, b in self.bases.items():
                                    if seq[z+m] == n:
                                        _motifs_tp1[g][l][b][m] += np.exp(Q[g][z])

                x += 1 # Incr seq count
            
            # Noramlize models to sum to 1
            _gamma_tp1 /= np.sum(_gamma_tp1, axis = 0)
            for g in groups:
                _lambda_tp1[g] /= np.sum(_lambda_tp1[g])
                for m in range(len(_motifs_tp1[g])):
                    for c in range(_motifs_tp1[g][m].shape[1]):
                        _motifs_tp1[g][m][:,c] /= np.sum(_motifs_tp1[g][m][:,c])
                        if self.kl_weights:
                            _motifs_tp1[g][m][:,c] *= np.sum(kl_div(_motifs_tp1[g][m][:,c], _background))
                            _motifs_tp1[g][m][:,c] /= np.sum(_motifs_tp1[g][m][:,c])


            # Check Convergence
            diff = np.abs(np.array((_gamma_tp1 - _gamma)))
            if iterations > self.min_iterations and bool(np.all(diff < self.epsilon)):
                converged = True

            _gamma = _gamma_tp1
            _lambda = _lambda_tp1
            _motifs = _motifs_tp1
            iterations += 1



        final_lens = np.argmax(_lambda, axis = 1)
        seq_ids = list(self.sequences.keys())

        models = {"Motif": _motifs, "Gamma": _gamma, "Lambda": _lambda, "Background": _background}
        results = {"Groups": {}, "Background": _background}

        for g in groups:
            results["Groups"][g] = {"Motif": _motifs[g][final_lens[g]]}

        return results, models


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
                    log.info("Initialization " + str(i) + ".")
                    results, models = self.__discover_EM(_gamma, _lambda, _motifs, _background, True)
                
                else:
                    print("ERROR: Algorithm " + algo + " is not implemented yet.")
                    sys.exit(1)
                
                classifications = self.classify_seq(self.sequences, results["Groups"], models["Background"])
                iter_LL = sum([float(classifications[header]["LL_Group_" + str(g)])/self.motif_ll(seq, np.array([[b] for b in models["Background"]]), models["Background"]) for header, seq in self.sequences.items() for g in range(self.num_groups)])
                LL_dict[iter_LL] = models
            
            log.info("Beinning Motif Discovery.")
            best_iter = LL_dict[max(LL_dict.keys())]
            results, models = self.__discover_EM(best_iter["Gamma"], best_iter["Lambda"], best_iter["Motif"], best_iter["Background"], False)
        
        else:
            _motifs, _lambda, _gamma, _background = self.__initialize_models(a = 100)
            results, models = self.__discover_EM(_gamma, _lambda, _motifs, _background, False)

        # if true, store results in Goops object
        if store:
            self.results = results

        return results




