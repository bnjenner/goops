import sys
import argparse
import random
import numpy as np

def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-n", "--num-seq", type=int, default=1000, help="Minimum motif length")
    parser.add_argument("-l", "--length", type=int, default=50, help="Seqeunce length")
    args = parser.parse_args()

    motif_list = ["TTTTTTTT",
                  "TTTTTTTT",
                  "TTTTTTTT",
                  "AAAAAAAA",
                  "AAAAAAAA",
                  "AAAAAAAA",
                  "AAAAAAAA",
                  "AAAAAAAA",
                  "AAAAAAAA",
                  "AAAAAAAA"]

    # motif_list = ["TTTTTTTT",
    #               "AAAAAAAA"]
    # motif_list = ["TGACTCA", # AP-1
    #               "CACGTGT"] # bHLH-ish

    #     A    C    G    T
    bg = [0.2, 0.3, 0.3, 0.2]

    n = len(motif_list)
    for i in range(args.num_seq):

        # Choose Motif and position
        motif = motif_list[i % n]

        l = len(motif)
        motif_pos = random.randint(0, (args.length - l)-1)

        # Generate random sequence around motif
        prefix = "".join(np.random.choice(["A","C","G","T"], p = bg, size = motif_pos))
        suffix = "".join(np.random.choice(["A","C","G","T"], p = bg, size = (args.length - (motif_pos + l))))

        # Print sequence as fasta
        header = ">seq_" + str(i) + "_" + motif
        seq = prefix + motif + suffix
        print(header, seq, sep = "\n")


if __name__ == "__main__":
    main()
    