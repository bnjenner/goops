import sys
import argparse
import random

def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-n", "--num-seq", type=int, default=100, help="Minimum motif length")
    parser.add_argument("-l", "--length", type=int, default=50, help="Seqeunce length")
    args = parser.parse_args()

    motif_list = ["TTTTTTTTTT",
    			  "AAAAAAAAAA"]

    for i in range(args.num_seq):

    	# Choose Motif and position
    	motif = random.choice(motif_list)
    	l = len(motif)
    	motif_pos = random.randint(0, (args.length - l)-1)

    	# Generate random sequence around motif
    	prefix = "".join(random.choices(["A","C","G","T"], k = motif_pos))
    	suffix = "".join(random.choices(["A","C","G","T"], k = (args.length - (motif_pos + l))))

    	# Print sequence as fasta
    	header = ">seq_" + str(i)
    	seq = prefix + motif + suffix
    	print(header, seq, sep = "\n")


if __name__ == "__main__":
    main()