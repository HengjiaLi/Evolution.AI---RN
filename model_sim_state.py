""""RN trained on the staet description dataset"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        #torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))
        model_save_name = 'sim_statenoGT.pth'
        path = F"/content/drive/My Drive/Evolution.AI---RN/{model_save_name}" 
        
        torch.save(self.state_dict(), path)


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.relation_type = args.relation_type
        
        self.g_dim = 256
        self.g_fc1 = nn.Linear(10*2+19, self.g_dim)
        self.g_fc2 = nn.Linear(self.g_dim, self.g_dim)
        self.g_fc3 = nn.Linear(self.g_dim, self.g_dim)
        self.g_fc4 = nn.Linear(self.g_dim, self.g_dim)

        self.f_fc1 = nn.Linear(self.g_dim, self.g_dim)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i//5-2)/2., (i%5-2)/2.]
            #return [(i//5-2)/2.,np.random.rand()]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))#64x25x2
        for i in range(25):#5x5 feature map
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = img ## x = (bs x 60)
        
        """g"""
        mb = x.size()[0]#mini batch 64
        n_channels = 10
        d = 6
        # x_flat = (64 x 6 x 10)
        x_flat = x.view(mb,d,n_channels)
        
        # add coordinates
        # x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        if self.relation_type != 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, d, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x6x10)
            x_i = x_i.repeat(1, d, 1, 1)  # (64x6x6x10)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x6x1x10)
            x_j = torch.cat([x_j, qst], 3)# (64x6x1x29)
            x_j = x_j.repeat(1, 1, d, 1)  # (64x6x6x29)
            
            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x6x6x(29+10))
        
            # reshape for passing through network
            x_ = x_full.view(mb * 6*6, n_channels*2+19)  # (64*6*6,39)
            #print(x_.size()[0])
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb, 6*6, self.g_dim)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 19, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)
