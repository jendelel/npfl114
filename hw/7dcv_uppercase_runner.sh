#!/bin/sh

set -e

for rnn in LSTM GRU
do
    for embedding in 200 250 300
    do
        for dim in 80 85 90 95 100
        do
            python 7dcv_uppercase-letters.py --rnn_cell=$rnn --rnn_cell_dim=$dim --embedding=$embedding --threads=8
        done
    done
done
