#!/bin/bash
cd ../../

python FedAvg.py --dataset 'mnist' \
--nclass 2 \
--nsamples 100 \
--nusers 100 \
--frac 0.1 \
--local_ep 5 \
--local_bs 32 \
--lr 0.1 \
--gpu 1 > ./log/FedLPS_mnist.txt
