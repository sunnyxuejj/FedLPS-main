#!/bin/bash
cd ../

nohup python3 FedLPS.py --dataset 'cifar10' \
--nclass 2 \
--nsamples 100 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--lr 0.1 \
--gpu 0 > ./log/FedLPS_cifar10.txt
