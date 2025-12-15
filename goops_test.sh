#!/bin/bash

call="goops -o ./goops_test \
	    -g 2 -m 8 -M 8 \
	    -a "EM" -i 20 \
	    ./test/small_test.fasta"
echo $call
eval $call
