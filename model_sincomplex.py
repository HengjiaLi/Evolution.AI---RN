#simple RN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
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
        model_save_name = 'sin_complex.pth'
        path = F"/content/drive/My Drive/Evolution.AI---RN/model_saved/{model_save_name}" 
        torch.save(self.state_dict(), path)


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        print('sin complex: product')
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24)*2+19, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        # define complex PE
        self.temperature = 10000
        # prepare coord tensor
        self.feature_dim = 24
        self.xy_dim = int(self.feature_dim/2)
        if args.cuda:
            #self.coord_tensor = self.coord_tensor.cuda()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.fcout = FCOutputModel()
        
        # shared-weights:same pixel location share same weights
        self.w_x = nn.Parameter(torch.rand(25),requires_grad=True).cuda() #25 for row
        self.cita_x = nn.Parameter(torch.rand(25),requires_grad=True).cuda() 
        self.r_x = nn.Parameter(torch.rand(25),requires_grad=True).cuda()
        
        self.w_y = nn.Parameter(torch.rand(25),requires_grad=True).cuda()  #25 for col
        self.cita_y = nn.Parameter(torch.rand(25),requires_grad=True).cuda() 
        self.r_y = nn.Parameter(torch.rand(25),requires_grad=True).cuda()
        
        # create a map for row/col indeces
        self.bs = args.batch_size
        self.row_pos = torch.arange(5.0).repeat_interleave(5)
        self.row_pos = torch.unsqueeze(self.row_pos,1).repeat(1,12).cuda()
        self.col_pos = torch.arange(5.0).repeat(5)#25x12
        self.col_pos = torch.unsqueeze(self.col_pos,1).repeat(1,12).cuda()
        
        self.x_pos = torch.FloatTensor( 25, self.xy_dim).cuda()#row
        self.y_pos = torch.FloatTensor( 25, self.xy_dim).cuda()#col
        self.pos = torch.cat((self.x_pos,self.y_pos),dim=1)#25x24
        self.pos = torch.unsqueeze(self.pos,0).repeat(self.bs,1,1)#64x25x24
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
    
    def cvt_coord_shared_weights(self):# compute complex embedding
        temp_wx = torch.unsqueeze(self.w_x,1).repeat(1,self.xy_dim)#25x12
        temp_citax = torch.unsqueeze(self.cita_x,1).repeat(1,self.xy_dim)#25x12
        temp_x = torch.mul(self.row_pos,temp_wx)+temp_citax
        self.x_pos = torch.stack((torch.sin(temp_x[:,0:self.xy_dim:2]),torch.cos(temp_x[:,1:self.xy_dim:2])),dim = 2).flatten(1)

        
        temp_wy = torch.unsqueeze(self.w_y,1).repeat(1,self.xy_dim)#25x12
        temp_citay = torch.unsqueeze(self.cita_y,1).repeat(1,self.xy_dim)#25x12
        temp_y = torch.mul(self.col_pos,temp_wy)+temp_citay
        self.y_pos=torch.stack((torch.sin(temp_y[:,0:self.xy_dim:2]),torch.cos(temp_y[:,1:self.xy_dim:2])),dim = 2).flatten(1)
        self.pos = torch.unsqueeze(torch.cat((self.x_pos,self.y_pos),dim=1),0).repeat(self.bs,1,1)#64x25x24

    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 (num of feature maps) x 5 x 5)
        """g"""
        mb = x.size()[0]#mini batch
        n_channels = x.size()[1]
        d = x.size()[2]
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1) # x_flat = (64 x 25 x 24)

        self.cvt_coord_shared_weights()
        x_flat =torch.mul(x_flat,self.pos)
        if self.relation_type != 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x24)
            x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x24)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x24)
            x_j = torch.cat([x_j, qst], 3)# (64x25x1x24+19)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x24+19)
            
            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*24+19)
        
            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d), 67)  # (64*25*25x(2*24+19)) = (40.000, 67)
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb, (d * d) * (d * d), 256)

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
