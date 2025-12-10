#!/bin/bash
# python main.py -m bilstm --n_layer 1 --hidden_dim 256 --epochs 20 --device cuda:7 --exp_name Exp1_BiLSTM_L1_H256
python main.py -m seq2seq --n_layer 1 --hidden_dim 256 --epochs 20 --device cuda:7 --exp_name Exp2_Seq2Seq_L1_H256
python main.py -m seq2seq --n_layer 2 --hidden_dim 256 --epochs 20 --device cuda:7 --exp_name Exp3_Seq2Seq_L2_H256
python main.py -m seq2seq --n_layer 2 --hidden_dim 512 --epochs 20 --device cuda:7 --exp_name Exp4_Seq2Seq_L2_H512
python main.py -m seq2seq --n_layer 2 --hidden_dim 256 --epochs 20 --device cuda:2 --exp_name Exp5_Seq2Seq_Attn_L2_H256