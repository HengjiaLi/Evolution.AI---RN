"""""""""
Pytorch implementation of RN trained/tested on the state description dataset
"""""""""
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np
import csv

os.environ["CUDA_CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model_sim_state import RN, CNN_MLP
from matplotlib import pyplot as plt
# import EarlyStopping
from pytorchtools import EarlyStopping

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

summary_writer = SummaryWriter()

if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
else:
  model = RN(args)
  
model_dirs = './model'
bs = args.batch_size
input_img = torch.FloatTensor(bs, 6*8)
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


def cvt_data_axis(data):
    img_states = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    img =[e[3] for e in data]
    return (img_states,qst,ans,img)
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
    
def train(epoch,rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_rels = []
    acc_norels = []

    l_binary = []
    l_unary = []

    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_rel,
                   accuracy_norel))
        
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    summary_writer.add_scalars('Accuracy/train', {
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    summary_writer.add_scalars('Loss/train', {
        'binary': avg_loss_binary,
        'unary': avg_loss_unary
    }, epoch)

    # return average accuracy
    return avg_acc_binary, avg_acc_unary,avg_loss_binary,avg_loss_unary

def test(epoch,rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    random.shuffle(rel)
    random.shuffle(norel)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []

    loss_binary = []
    loss_unary = []
    
    #print(len(rel[0]))
    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())
        #print(l_bin.item())
        #print(batch_idx)
    
    avg_acc_binary = sum(accuracy_rels) / len(accuracy_rels)
    avg_acc_unary = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(avg_acc_binary, avg_acc_unary))

    summary_writer.add_scalars('Accuracy/test', {
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    avg_loss_binary = sum(loss_binary) / len(loss_binary)
    avg_loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/test', {
        'binary': avg_loss_binary ,
        'unary': avg_loss_unary
    }, epoch)
    
    #sanity check
    #
    # if avg_loss_binary>1:
    #     print(loss_binary)
    #     model_save_name = 'overflow.pth'
    #     path = F"/content/drive/My Drive/Evolution.AI---RN/{model_save_name}" 
    #     torch.save(model.state_dict(), path)
    return avg_acc_binary, avg_acc_unary,avg_loss_binary,avg_loss_unary

def validate(epoch,rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []

    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())
    
    avg_acc_binary = sum(accuracy_rels) / len(accuracy_rels)
    avg_acc_unary = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Validate set: Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(avg_acc_binary, avg_acc_unary))

    summary_writer.add_scalars('Accuracy/val', {
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    avg_loss_binary = sum(loss_binary) / len(loss_binary)
    avg_loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/val', {
        'binary': avg_loss_binary ,
        'unary': avg_loss_unary
    }, epoch)

    return avg_acc_binary, avg_acc_unary,avg_loss_binary,avg_loss_unary
def test_closest(data):
    model.eval()
    accuracy = []
    for batch_idx in range(len(data[0]) // bs):
        tensor_data(data, batch_idx)
        acc_bin, l = model.test_(input_img, input_qst, label)
        accuracy.append(acc_bin.item())
    acc = sum(accuracy) / len(accuracy)
    print('\n Q4 accuracy: {:.0f}% \n'.format(acc))
    return acc
    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'more-clevr_state.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets, val_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    rel_val = []
    norel_train = []
    norel_test = []
    norel_val = []
    print('processing data...')

    for img_states, relations, norelations,img in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img_states,qst,ans,img))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img_states,qst,ans,img))

    for img_states, relations, norelations,img in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img_states,qst,ans,img))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img_states,qst,ans,img))
            
    for img_states, relations, norelations,img in val_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_val.append((img_states,qst,ans,img))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_val.append((img_states,qst,ans,img))
    
    return (rel_train, rel_test,rel_val, norel_train, norel_test,norel_val)
    
if __name__ == "__main__":
    rel_train, rel_test,rel_val,norel_train, norel_test,norel_val = load_data()
    _,_,_,Q4,Q5,Q6,Q7,Q8 = split_data(rel_test)
    try:
        os.makedirs(model_dirs)
    except:
        print('directory {} already exists'.format(model_dirs))

    if args.resume:
        filename = os.path.join(model_dirs, args.resume)
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {}'.format(filename))
    #print(list(model.parameters()))
    train_acc_binary_history= []
    train_acc_unary_history = []
    train_loss_binary_history = []
    train_loss_unary_history = []
    
    test_acc_binary_history= []
    test_acc_unary_history = []
    test_loss_binary_history = []
    test_loss_unary_history = []
    
    val_acc_binary_history= []
    val_acc_unary_history = []
    val_loss_binary_history = []
    val_loss_unary_history = []
    patience = 5
    with open(f'./{args.model}_{args.seed}_log.csv', 'w') as log_file:
        csv_writer = csv.writer(log_file, delimiter=',')
        csv_writer.writerow(['epoch', 'train_acc_rel',
                        'train_acc_norel', 'test_acc_rel', 'test_acc_norel'])

        print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")
        
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(1, args.epochs + 1):
            train_acc_binary, train_acc_unary,train_loss_binary,train_loss_unary = train(
                epoch, rel_train, norel_train)
            
            val_acc_binary, val_acc_unary,val_loss_binary,val_loss_unary = validate(
                epoch, rel_val, norel_val)
                
            test_acc_binary, test_acc_unary,test_loss_binary,test_loss_unary = test(
                epoch, rel_test, norel_test)
            acc = test_closest(Q4)
            
            train_acc_binary_history.append(train_acc_binary)
            train_acc_unary_history.append(train_acc_unary)
            train_loss_binary_history.append(train_loss_binary)
            train_loss_unary_history.append(train_loss_unary)
            # if test_loss_binary>1:
            #     test_loss_binary = test_loss_binary_history[-1]
            #     test_loss_unary = test_loss_unary_history[-1]
            test_acc_binary_history.append(test_acc_binary)
            test_acc_unary_history.append(test_acc_unary)
            test_loss_binary_history.append(test_loss_binary)
            test_loss_unary_history.append(test_loss_unary)
            
            val_acc_binary_history.append(val_acc_binary)
            val_acc_unary_history.append(val_acc_unary)
            val_loss_binary_history.append(val_loss_binary)
            val_loss_unary_history.append(val_loss_unary)

            csv_writer.writerow([epoch, train_acc_binary,
                            train_acc_unary, test_acc_binary, test_acc_unary])
                            
            early_stopping(val_loss_binary+val_loss_unary, model)
            if early_stopping.early_stop:
                print('early stopped')
                break
        final_epoch = epoch
        model.load_state_dict(torch.load('checkpoint.pt'))
        model.save_model(epoch)
        
    plt.figure()
    epo = range(1, final_epoch + 1)
    plt.plot(epo,train_loss_binary_history)
    plt.plot(epo,train_loss_unary_history)
    plt.plot(epo,test_loss_binary_history)
    plt.plot(epo,test_loss_unary_history)
    plt.axvline(x=final_epoch-patience,color='r', linestyle='--')
    plt.legend(['train_loss_binary','train_loss_unary','test_loss_binary','test_loss_unary','early_stop'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1,final_epoch + 1,5))
    plt.show()
    FIGname = 'train_curve_state.png'
    FIGpath = F"/content/drive/My Drive/Evolution.AI---RN/{FIGname}"
    plt.savefig(FIGpath)
    
    # # remove dataset
    # data_dir = "./data/more-clevr.pickle"
    # if os.path.isfile(data_dir):
    #     os.remove(data_dir)
    # else:    ## Show an error ##
    #     print("Error: %s file not found" % data_dir)
