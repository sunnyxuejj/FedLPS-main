#!/bin/bash
cd ../../

nohup python3 FedAvg.py --dataset 'tinyimagenet' \
--nclass 20 \
--nsamples 40 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--lr 0.1 \
--gpu 1 > ./log/FedAvg_tinyimagenet.txt
