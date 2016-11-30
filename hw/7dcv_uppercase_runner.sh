#!/bin/sh

for rnn in LSTM GRU
do
    for embedding in -1 50 100 150 200
    do
        for dim in 20 30 40 50
        do
            python 7dcv_uppercase-letters.py --rnn_cell=$rnn --rnn_cell_dim=$dim --embedding=$embedding --threads=4 
        done
    done
done
