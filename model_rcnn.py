#simple RN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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
        model_save_name = 'rn_rcnn.pth'
        path = F"/content/drive/My Drive/Evolution.AI---RN/model_saved/{model_save_name}" 
        
        torch.save(self.state_dict(), path)


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        #print('sim')
        #self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        
        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_save_name = 'rcnn.pth'
        path = F"/content/drive/My Drive/Evolution.AI---RN/model_saved/{model_save_name}" 
        self.RCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 12+1
        in_features = self.RCNN.roi_heads.box_predictor.cls_score.in_features
        self.RCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.RCNN.load_state_dict(torch.load(path))
        
        for param in self.RCNN.parameters():
            param.requires_grad = False
        #model.to(device)
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_dim = 256
        self.g_fc1 = nn.Linear(34+15, self.g_dim)
        self.g_fc2 = nn.Linear(self.g_dim, self.g_dim)
        self.g_fc3 = nn.Linear(self.g_dim, self.g_dim)
        self.g_fc4 = nn.Linear(self.g_dim, self.g_dim)

        self.f_fc1 = nn.Linear(self.g_dim, self.g_dim)

        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
    
        # and a learning rate scheduler
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=5,
                                                        gamma=0.1)


    def forward(self, img, qst):
        self.RCNN.eval()
        predictions = self.RCNN(img) ## img = (bs x 3x75x75)
        boxes =torch.stack([t['boxes'][:6] for t in predictions])
        centers = (boxes[:,:,:2]+boxes[:,:,2:])/2
        #centers
        labels = torch.stack([t['labels'][:6] for t in predictions])
        one_hot = torch.nn.functional.one_hot(labels,num_classes=13)
        
        x = torch.cat((centers,one_hot),dim=2)#x = bsx6x15
        """g"""
        #print(x.size())
        mb = x.size()[0]#bs
        n_channels = x.size()[2]
        d = x.size()[1]
        #print(d)
        #print(n_channels)
        # x_flat = (64 x 6 x 10)
        x_flat = x.view(mb,d,n_channels)#bsx6x15
        
        if self.relation_type != 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, d, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (bsx1x6x15)
            x_i = x_i.repeat(1, d, 1, 1)  # (bsx6x6x15)
            x_j = torch.unsqueeze(x_flat, 2)  # (bsx6x1x15
            x_j = torch.cat([x_j, qst], 3)# (bsx6x1x15+19)
            x_j = x_j.repeat(1, 1, d, 1)  # (bsx6x6x34)
            
            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x6x6x(34+15))
            #print(x_full.size())
            # reshape for passing through network
            x_ = x_full.view(mb * 6*6, 49)  # (64*6*6,39)
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
