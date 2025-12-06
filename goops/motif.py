import logging
import math
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
        self.sequeces = sequeces
        self.motifs = None


    ######################################################################
    # Initialize Motif Models
    def __initialize_models(self, min_length: int, max_length: int, num_groups: int):
        
        # Uniform Length and Group Probs
        num_lengths = (max_length - min_length + 1)
        g_prob = math.log(1 / num_groups)
        l_prob = math.log(1 / num_lengths)
        n_prob = math.log(0.1)
        bias_n_prob = math.log(0.7)

        # Initialize Matricies
        models, lengths, groups = [], [], []
       
        for i in range(num_groups):     
            groups.append(g_prob)
            lengths.append(np.full((1, num_lengths), l_prob))
            models.append([])

            # Initlize length motif model with nucleotide bias
            for l in range(min_length, max_length+1):
                models[i].append(np.full((4, l), n_prob))
                models[i][-1][i] = bias_n_prob

        return models, lengths, groups


    ######################################################################
    # Expectation Maximization Goops Implementation
    def __discover_EM(self, models: list):
        return


    ######################################################################
    # Metropolis-Hastings Goops Implementation
    def __discover_MH(self, models: list):
        return

    ######################################################################
    # Discover Motif Auxillary Functions
    def discover(self, min_length: int, max_length: int, algo: str = "EM"):

        print("Parameters:")
        print(" - Min-Length:", min_length)
        print(" - Max-Length:", max_length)
        print(" - Algorithm:", algo)

        models, lengths, groups = self.__initialize_models(min_length, max_length, num_groups = 2)
        
        # for i in range(2):
        #     print("Group:", i)
        #     print(groups[i])
        #     print(lengths[i])
        #     for m in models[i]:
        #         print(m)

        # # Run Aglorithm
        # if algo == "EM":
        #     self.__discover_EM(models)
        # elif algo == "MH":
        #     self.__discover_MH(models)
        # else:
        #     print("ERROR: Algorithm is not implemented yet.")
        #     sys.exit(1)



