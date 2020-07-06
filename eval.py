import os
import pickle
import numpy as np
import torch
import argparse
import random
from main import load_data,cvt_data_axis
from model_simnoPE import RN

rel_train, rel_test, norel_train, norel_test = load_data()
# SPLIT DATASET
norel_pos = []#non-relational questions that require positional info
norel_nopos = []#non-relational questions that does not require positional info
for i in range(len(norel_test)):
    if norel_test[i][1][9]==1 or norel_test[i][1][10]==1:
        norel_pos.append(norel_test[i])
    else:
        norel_nopos.append(norel_test[i])
random.shuffle(norel_pos)
random.shuffle(norel_nopos)
norel_pos = cvt_data_axis(norel_pos)
norel_nopos = cvt_data_axis(norel_nopos)
# LOAD MODEL
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
model = RN(args)
path="./All Model/simple_RNnoPE.pth"
if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
else:
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
print(np.shape(norel_pos))
print(np.shape(norel_nopos))
acc,l =model.test_(torch.FloatTensor(norel_pos[0]), torch.FloatTensor(norel_pos[1]), torch.LongTensor(norel_pos[2]))
print('\n Test set: Unary accuracy (need pos info): {:.0f}%\n'.format(acc))
