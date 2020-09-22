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
        return F.log_softmax(x - torch.max(x,1,keepdim=True)[0], dim=1)

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
        #print(loss.item())
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        #torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))
        model_save_name = 'sin_relative.pth'
        path = F"/content/drive/My Drive/Evolution.AI---RN/{model_save_name}" 
        
        torch.save(self.state_dict(), path)


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        print('relative + 4d sin/cos')
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear(48+10+19, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        # self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        # self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        # if args.cuda:
        #     self.coord_oi = self.coord_oi.cuda()
        #     self.coord_oj = self.coord_oj.cuda()
        # self.coord_oi = Variable(self.coord_oi)
        # self.coord_oj = Variable(self.coord_oj)

        self.temperature = 10000
        # prepare coord tensor
        self.feature_dim = 24
        self.xy_dim = int(self.feature_dim/2)
        def cvt_coord(i,j):
            dim_t = np.arange(self.xy_dim)
            dim_t = self.temperature**(2 * (dim_t // 2) / self.feature_dim )
            
            #positional encoding for pixel at ith position
            x1 = i//5#row
            y1 = i%5#col
            x1_pos = dim_t.copy()
            y1_pos = dim_t.copy()
            
            x1_pos[0:self.xy_dim:2] = np.sin(x1/x1_pos[0:self.xy_dim:2])
            x1_pos[1:self.xy_dim:2] = np.cos(x1/x1_pos[1:self.xy_dim:2])
            y1_pos[0:self.xy_dim:2] = np.sin(y1/y1_pos[0:self.xy_dim:2])
            y1_pos[1:self.xy_dim:2] = np.cos(y1/y1_pos[1:self.xy_dim:2])
            #y_pos = np.random.rand(self.xy_dim)
            pos1 = np.concatenate((x1_pos[:2],y1_pos[:2]))
            
            #positional encoding for pixel at ith position
            x2 = j//5#row
            y2 = j%5#col
            x2_pos = dim_t.copy()
            y2_pos = dim_t.copy()
            
            x2_pos[0:self.xy_dim:2] = np.sin(x2/x2_pos[0:self.xy_dim:2])
            x2_pos[1:self.xy_dim:2] = np.cos(x2/x2_pos[1:self.xy_dim:2])
            y2_pos[0:self.xy_dim:2] = np.sin(y2/y2_pos[0:self.xy_dim:2])
            y2_pos[1:self.xy_dim:2] = np.cos(y2/y2_pos[1:self.xy_dim:2])
            #y_pos = np.random.rand(self.xy_dim)
            pos2 = np.concatenate((x2_pos[:2],y2_pos[:2]))
            
            x_relative = np.dot(x1_pos,x2_pos)
            y_relative = np.dot(y1_pos,y2_pos)
    
            pos = np.concatenate([pos1,pos2,[x_relative,y_relative]])
            #pos = [x_relative,y_relative]
            return pos
        #self.coord_tensor = torch.FloatTensor(args.batch_size,25,25,self.feature_dim*2+2)
        self.coord_tensor = torch.FloatTensor(args.batch_size,25,25,10)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        #np_coord_tensor = np.zeros((args.batch_size, 25, 25, self.feature_dim*2+2))#64x25x25x(24*2+2)
        np_coord_tensor = np.zeros((args.batch_size, 25, 25,10))#64x25x25x(2)
        for i in range(25):#5x5 feature map
            for j in range(25):
                np_coord_tensor[:,i,j,:] = np.array( cvt_coord(i,j) )
        #print(np_coord_tensor[0,0,:,-2:])
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))#64x25x25x(24*2+2)

        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 (num of feature maps) x 5 x 5)
        """g"""
        mb = x.size()[0]#mini batch
        n_channels = x.size()[1]
        d = x.size()[2]
        
        #x_flat = torch.einsum('abcd->abdc', x)#change flatten direction to col by col
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        #x_flat1 = x.view(mb,n_channels,d*d).permute(0,2,1)
        #x_flat2 = x_flat1.clone()
        #x_flat = torch.cat([x_flat1, x_flat2],2)
        
        if self.relation_type != 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x24+19)
            x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x24+19)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x24+19)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x24+19)
            
            # concatenate all together
            x_notfull = torch.cat([x_i,x_j],3) # (64x25x25x48+19)
            
            # add coordinates
            x_full = torch.cat([x_notfull, self.coord_tensor],3)# (64x25x25x(48+50+19))
            #x_flat =x_flat+self.coord_tensor
            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d), 58+19)  # (64*25*25x(50+48+19)) = (40.000, 117)
            
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
