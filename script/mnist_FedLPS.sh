#!/bin/bash
cd ../

nohup python3 FedLPS.py --dataset 'mnist' \
--nclass 2 \
--nsamples 100 \
--nusers 100 \
--frac 0.1 \
--local_ep 5 \
--local_bs 32 \
--online_decision True \
--lr 0.1 > ./log/FedLPS_mnist.txt
