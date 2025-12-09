import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#sys.path.insert(0, '/Users/ardenlee/Code/goops') 
from goops import utils

"""
argv[1] = motif model 1
argv[2] = motif model 2
argv[3] = fasta 
"""

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

    #mm1.to_csv('mm1.csv', index=True)
    #mm2.to_csv('mm2.csv', index=True)

# returns a list of probs of starting at each pos for one motif model
def prob_motif_along_seq_for_mmX(seq, mmX):
    """
    calc prob ( seq_window | motif ) at each start pos
    along the sequence
    """
    len_motif = len(mmX.columns)
    ind_list = mmX.index.tolist()
    
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
            prob_nt_at_pos_x = mmX.iloc[nt_index, motif_pos]

            prob_motif_here *= prob_nt_at_pos_x #### there is an issue here somewhere

        catch_prob_motif_at_each_pos.append(prob_motif_here)

    return catch_prob_motif_at_each_pos

mm1_path = sys.argv[1]
mm2_path = sys.argv[2]
fasta_path = sys.argv[3]

mm1 = pd.read_csv(mm1_path, index_col=0)
mm2 = pd.read_csv(mm2_path, index_col=0)
seq_dict = utils.read_fasta(fasta_path)

seq_keys = list(seq_dict.keys())
print(seq_keys)
for i in seq_keys[:3]:   ############ this will just iterate through all seqs
    seq = seq_dict[i]

    mm1_in_seq = prob_motif_along_seq_for_mmX(seq, mm1)
    mm2_in_seq = prob_motif_along_seq_for_mmX(seq, mm2)


    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

    ax1.plot(mm1_in_seq)
    ax1.set_xlabel("Position Along Sequence")
    ax1.set_ylabel("Likelihood of Motif")
    ax1.set_title("Motif A")

    ax2.plot(mm2_in_seq)
    ax2.set_xlabel("Position Along Sequence")
    ax2.set_title("Motif B")
    
    i = i.strip(">")
    plt.suptitle("Likelihood of Motif A and B Across Sequence: '" + i + "'")
    plt.show()