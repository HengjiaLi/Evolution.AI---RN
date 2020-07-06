import os
import pickle
import numpy as np
import torch
import argparse
import random
from main import load_data,cvt_data_axis
from model_simnoPE import RN

# SPLIT DATASET
def split_data(data,type_):
    pos = []#questions that require positional info
    nopos = []#questions that does not require positional info
    if type_ == "norel":
        for i in range(len(data)):
            if data[i][1][9]==1 or data[i][1][10]==1:
                pos.append(data[i])
            else:
                nopos.append(data[i])
    elif type_ == "rel":
        for i in range(len(data)):
            if data[i][1][8]==1 or data[i][1][9]==1:
                pos.append(data[i])
            else:
                nopos.append(data[i])

    random.shuffle(pos)
    random.shuffle(nopos)
    pos = cvt_data_axis(pos)
    nopos = cvt_data_axis(nopos)
    return pos,nopos

# Model setup
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
#path="./All Model/simple_RNnoPE.pth"
path="./All Model/simple_RN.pth"
if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
else:
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


# TEST
rel_train, rel_test, norel_train, norel_test = load_data()
norel_pos,norel_nopos = split_data(norel_test,'norel')
acc,l =model.test_(torch.FloatTensor(norel_pos[0]), torch.FloatTensor(norel_pos[1]), torch.LongTensor(norel_pos[2]))
print('\n Test set: Unary accuracy (need pos info): {:.0f}%\n'.format(acc))
acc,l =model.test_(torch.FloatTensor(norel_nopos[0]), torch.FloatTensor(norel_nopos[1]), torch.LongTensor(norel_nopos[2]))
print('\n Test set: Unary accuracy (need no pos info): {:.0f}%\n'.format(acc))

rel_pos,rel_nopos = split_data(rel_test,'rel')
acc,l =model.test_(torch.FloatTensor(rel_pos[0]), torch.FloatTensor(rel_pos[1]), torch.LongTensor(rel_pos[2]))
print('\n Test set: Binary accuracy (need pos info): {:.0f}%\n'.format(acc))
acc,l =model.test_(torch.FloatTensor(rel_nopos[0]), torch.FloatTensor(rel_nopos[1]), torch.LongTensor(rel_nopos[2]))
print('\n Test set: Binary accuracy (need no pos info): {:.0f}%\n'.format(acc))

