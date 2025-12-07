import logging
import math
import numpy as np
from scipy.special import logsumexp
from goops import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

#########################################################################################################
# Motif Discovery Class
class Goops:

    ######################################################################
    # Constructor
    def __init__(self, sequeces: dict):
        self.sequeces = sequeces
        self.motifs = None
        self.pseudocount = 0.001


    ######################################################################
    # Initialize Motif Models
    def __initialize_models(self, min_length: int, max_length: int, num_groups: int, init: bool = False):
        
        num_lengths = (max_length - min_length + 1)

        if init:
            # Uniform Length and Group Probs
            g_prob = math.log(1 / num_groups)
            l_prob = math.log(1 / num_lengths)
            n_prob = math.log(0.1)
            bias_n_prob = math.log(0.7)
        else:
            # Pseudocount (basically zero)
            g_prob = math.log(self.pseudocount)
            l_prob = math.log(self.pseudocount)
            n_prob = math.log(self.pseudocount)

        # Initialize Matricies 
        _gamma = np.full(num_groups, g_prob, dtype=np.float64)
        _lambda = [np.full(num_lengths, l_prob, dtype=np.float64) for g in range(num_groups)]
        _models = [[] for g in range(num_groups)]
        
        # Initlize length motif model with nucleotide bias
        for i in range(num_groups):
            for l in range(min_length, max_length+1):
                _models[i].append(np.full((4, l), n_prob, dtype=np.float64))
                if init:
                    _models[i][-1][i,:] = bias_n_prob

        return _models, _lambda, _gamma


    ######################################################################
    # Expectation Maximization Goops Implementation
    def __discover_EM(self, _models: list, _lambda: list, _gamma: np.ndarray):

        bases = {"A": 0, "C": 1, "G": 2, "T": 3}
        num_groups = len(_gamma)
        num_lengths = _lambda[0].shape[0]
        lengths = [m.shape[1] for m in _models[0]]
        groups = [g for g in range(len(_gamma))]

        group_Q = np.full(num_groups, math.log(self.pseudocount), dtype=np.float64)
        _models_tp1, _lambda_tp1, _gamma_tp1 = self.__initialize_models(lengths[0], 
                                                                        lengths[-1], 
                                                                        num_groups = 2,
                                                                        init = False)


        ITERATIONS = 0
        while ITERATIONS < 100:

            print("Iteration:", ITERATIONS)

            # Iterate through sequences
            for header, seq in self.sequeces.items():

                for l in range(num_lengths):
                    last_pos = len(seq) - lengths[l] + 1 # Also number of possible positions
                    Q = np.array([group_Q.copy() for q in range(last_pos)], dtype=object)

                    # Iterate through groups and all possible start positions            
                    for g in groups:                    
                        for i in range(last_pos):
                            end_pos = i + lengths[l] - 1 # motif end pos

                            for x in range(len(seq)):
                                b = bases[seq[x]]
                                if x >= i and x <= end_pos:
                                    Q[i][g] += _models[g][l][b][x - i]
                                else:
                                    Q[i][g] += math.log(0.25) # Uniform Background
                                Q[i][g] += _gamma[g] + _lambda[g][l] + math.log(1 / last_pos)


                            _gamma_tp1[g] = logsumexp([_gamma_tp1[g], Q[i][g]])
                            _lambda_tp1[g][l] = logsumexp([_lambda_tp1[g][l], Q[i][g]])

                            for m in range(lengths[l]):
                                for n, b in bases.items():
                                    if seq[i+m] == n: 
                                        _models_tp1[g][l][b][m] += logsumexp([_models_tp1[g][l][b][m], Q[i][g]])


            # Noramlize models to sum to 1
            _gamma_tp1 = utils.logsafe_normalize(_gamma_tp1)
            for g in groups:
                _lambda_tp1[g] = utils.logsafe_normalize(_lambda_tp1[g])
                for m in range(len(_models_tp1[g])):
                    for c in range(_models_tp1[g][m].shape[1]):
                        _models_tp1[g][m][:,c] = utils.logsafe_normalize(_models_tp1[g][m][:,c]) 

            _gamma = _gamma_tp1
            _lambda = _lambda_tp1
            _models = _models_tp1
            ITERATIONS += 1


        print(np.exp(_gamma))
        print(np.exp(_gamma_tp1))
        cont = input("TEST")
        for l in _lambda:
            print(np.exp(l))
        for l in _lambda_tp1:
            print(np.exp(l))
        for m in _models_tp1[0]:
            print(np.exp(m))

        return


    ######################################################################
    # Metropolis-Hastings Goops Implementation
    def __discover_MH(self, models: list):
        return

    ######################################################################
    # Discover Motif Auxillary Functions
    def discover(self, min_length: int, max_length: int, algo: str = "EM"):

        _models, _lambda, _gamma = self.__initialize_models(min_length, 
                                                            max_length,
                                                            num_groups = 2,
                                                            init = True)

        print("Parameters:")
        print(" - Min-Length:", min_length)
        print(" - Max-Length:", max_length)
        print(" - Algorithm:", algo)
        print(" - _models:", len(_models), len(_models[0]), _models[0][0].shape)
        print(" - _lambda:", len(_lambda), _lambda[0].shape)
        print(" - _gamma:", len(_gamma))

        # Run Aglorithm
        if algo == "EM":
            self.__discover_EM(_models, _lambda, _gamma)
        elif algo == "MH":
            self.__discover_MH(models)
        else:
            print("ERROR: Algorithm is not implemented yet.")
            sys.exit(1)



