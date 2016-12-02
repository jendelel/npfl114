#!/bin/sh

set -e

for rnn in LSTM GRU
do
    for embedding in -1 100 150 200
    do
        for dim in 30 40 50
        do
            for dropout in 0.5 0.8 1.0
            do
                python 7dcv_uppercase-letters.py --rnn_cell=$rnn --rnn_cell_dim=$dim --embedding=$embedding --threads=8 --dropout=$dropout
            done
        done
    done
done
