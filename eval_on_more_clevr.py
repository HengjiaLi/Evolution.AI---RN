import os
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import argparse
import random
from main import load_data,cvt_data_axis
from model_sin import RN

# SPLIT DATASET
def split_data(data):
    Q1 = []#shape of object
    Q2 = []#query vertical position->yes/no
    Q3 = []#query horizontal position->yes/no
    Q4 = []#closest-to->rectangle/circle
    Q5 = []#furthest-from->rectangle/circle
    Q6 = []#count same color->1~6
    Q7 = []#if color2 is above the current object --->yes or no
    Q8 = []#if color2 is on the left to the current object --->yes or no
    random.shuffle(data)
    for ind,sample in enumerate(data):
        Q = sample[1]
        if Q[12] == 1: #non-rel questions
            if Q[14] == 1:
                Q1.append(sample)
            elif Q[15] == 1:
                Q2.append(sample)
            elif Q[16] == 1:
                Q3.append(sample)
        elif Q[13] == 1:#rel questions
            if Q[14] == 1:
                Q4.append(sample)
            elif Q[15] == 1:
                Q5.append(sample)
            elif Q[16] == 1:
                Q6.append(sample)
            elif Q[17] == 1:
                Q7.append(sample)
            elif Q[18] == 1:
                Q8.append(sample)

    Q1 = cvt_data_axis(Q1)
    Q2 = cvt_data_axis(Q2)
    Q3 = cvt_data_axis(Q3)
    Q4 = cvt_data_axis(Q4)
    Q5 = cvt_data_axis(Q5)
    Q6 = cvt_data_axis(Q6)
    Q7 = cvt_data_axis(Q7)
    Q8 = cvt_data_axis(Q8)
    return Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8

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
#path="./model/epoch_RN_20.pth"
#path="./model/sim_noH.pth"
model_save_name = 'sin_summed.pth'
path = F"/content/drive/My Drive/Evolution.AI---RN/model_saved/{model_save_name}"#Gdrive path
if torch.cuda.is_available():
    # os.environ["CUDA_CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model.load_state_dict(torch.load(path))
else:
    print('useing CPU')
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 19)
label = torch.LongTensor(bs)
if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()
input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)

def test(data):
    model.eval()
    accuracy = []
    for batch_idx in range(len(data[0]) // bs):
        tensor_data(data, batch_idx)
        acc_bin, l = model.test_(input_img, input_qst, label)
        accuracy.append(acc_bin.item())
    acc = sum(accuracy) / len(accuracy)
    return acc
# TEST
rel_train, rel_test,rel_val,norel_train, norel_test,norel_val = load_data()
Q1,Q2,Q3,_,_,_,_,_ = split_data(norel_test)
acc = test(Q1)
print('\n Test set: Unary accuracy (shape of object): {:.0f}%\n'.format(acc))
acc = test(Q2)
print('\n Test set: Unary accuracy (query vertical position): {:.0f}%\n'.format(acc))
acc = test(Q3)
print('\n Test set: Unary accuracy (query horizontal position->yes/no): {:.0f}%\n'.format(acc))

_,_,_,Q4,Q5,Q6,Q7,Q8 = split_data(rel_test)
acc = test(Q4)
print('\n Test set: Binary accuracy (closest-to): {:.0f}%\n'.format(acc))
acc = test(Q5)
print('\n Test set: Binary accuracy (furthest-to): {:.0f}%\n'.format(acc))
acc = test(Q6)
print('\n Test set: Binary accuracy (count same color): {:.0f}%\n'.format(acc))
acc = test(Q7)
print('\n Test set: Binary accuracy (above object): {:.0f}%\n'.format(acc))
acc = test(Q8)
print('\n Test set: Binary accuracy (left to object): {:.0f}%\n'.format(acc))
