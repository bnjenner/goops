import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_mms_from_motifs():
    len_motif = 8
    motif_1 = "AAAAAAAA"
    motif_2 = "TTTTTTTT"

    ind_list = ["A", "T", "G", "C"]
    default =  0.01 #np.log(0.01) #0.01

    mm1 = pd.DataFrame(np.full((4,len_motif), default), columns = list(motif_1), index=ind_list)
    mm2 = pd.DataFrame(np.full((4,len_motif), default), columns = list(motif_2), index=ind_list)

    m_dict = {
        "m1" : [motif_1, mm1],
        "m2" : [motif_2, mm2]
    }

    for key in m_dict.keys():
        motif_x = m_dict[key][0]
        mm_x = m_dict[key][1]
        for pos in range(len_motif):
            nt_index = ind_list.index(motif_x[pos])
            mm_x.iloc[nt_index, pos] = 0.99 #np.log(0.99) #0.99

    print(mm1)
    print(mm2)


#fasta = sys.argv[1]
mm1_path = sys.argv[1]
mm2_path = sys.argv[2]


#mm1_path = "/Users/ardenlee/Code/outside_goops/mm1.csv"
#mm2_path = "/Users/ardenlee/Code/outside_goops/mm1.csv"
mm1 = pd.read_csv(mm1_path, index_col=0)
mm2 = pd.read_csv(mm2_path, index_col=0)

len_motif = len(mm1.columns)
ind_list = ["A", "T", "G", "C"]

# path = "/Users/ardenlee/Code/outside_goops/test_seq.fa"
# with open(path, "r") as p:

seq = "GAGTAAAGGATTACCTGTGCAAAAAAAAGACGCCGAGCGTCTACTCTGAC"






"""
calc prob ( seq_window | motif ) at each start pos
along the sequence
"""

# need motif model

catch_prob_motif_at_each_pos = []

# for each start position in seq
for start_pos in range(len(seq) - len_motif):
    window = seq[start_pos:start_pos+len_motif]
    print(window)
    prob_motif_here = 1

    # for each position in the motif
    for motif_pos in range(len_motif):
        
        nt_at_pos_x = window[motif_pos]
        nt_index = ind_list.index(nt_at_pos_x)

        # get probability that pos_x = nucleotide at x | mm1
        prob_nt_at_pos_x = mm1.iloc[nt_index, motif_pos]
        #print(prob_nt_at_pos_x)


        prob_motif_here *= prob_nt_at_pos_x #### there is an issue here somewhere


    catch_prob_motif_at_each_pos.append(prob_motif_here)

print(len(seq))
print(len(catch_prob_motif_at_each_pos))
print(catch_prob_motif_at_each_pos)


plt.plot(catch_prob_motif_at_each_pos)
plt.show()


#mm1.to_csv('mm1.csv', index=True)
#mm2.to_csv('mm2.csv', index=True)