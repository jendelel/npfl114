#!/bin/sh

for rnn in LSTM GRU
do
    for embedding in -1 50 150 250
    do
        for dim in 8 10 12
        do
            python 7dcv_uppercase-letters.py --rnn_cell=$rnn --rnn_cell_dim=$dim --embedding=$embedding --threads=4 
        done
    done
done
