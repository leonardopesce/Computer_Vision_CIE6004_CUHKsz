import torch
from networks.STN import STN
from networks.QSTN import QSTN
from torch import nn
import torch.nn.functional as F

class PointNetfeat(nn.Module):
    def __init__(self, num_points=500, point_tuple=1):
        super(PointNetfeat, self).__init__()
        self.num_points = num_points
        self.point_tuple = point_tuple

        self.stn1 = QSTN(num_points=num_points*self.point_tuple, dim=3)

        self.stn2 = STN(num_points=num_points, dim=64)

        self.bnca = BatchNormalizedConv1d(3*self.point_tuple, 64)
        self.bncb = BatchNormalizedConv1d(64, 64)
        self.bnc1 = BatchNormalizedConv1d(64, 64)
        self.bnc2 = BatchNormalizedConv1d(64, 128)
        self.bnc3 = BatchNormalizedConv1d(128, 1024)

        self.mp1 = torch.nn.MaxPool1d(num_points)

    def forward(self, x):

        # input transform
        # from tuples to list of single points
        x = x.view(x.size(0), 3, -1)
        trans = self.stn1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)

        # mlp (64,64)
        x = F.relu(self.bncb(F.relu(self.bnca(x))))

        # feature transform
        trans2 = self.stn2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2, 1)
        
        # mlp (64,128,1024)
        x = self.bnc3(F.relu(self.bnc2(F.relu(self.bnc1(x)))))

        # symmetric max operation over all points 
        # (not to much difference with sum according to the paper)
        x = self.mp1(x)

        x = x.view(-1, 1024)

        return x, trans, trans2, None
    

class PCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, point_tuple=1):
        super(PCPNet, self).__init__()
        self.num_points = num_points

        self.feat = PointNetfeat(num_points=num_points, point_tuple=point_tuple)
        self.bnfc1 = BatchNormalizedLinear(1024, 512)
        self.bnfc2 = BatchNormalizedLinear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = F.relu(self.bnfc1(x))
        x = self.do1(x)
        x = F.relu(self.bnfc2(x))
        x = self.do2(x)
        x = self.fc3(x)

        return x, trans, trans2, pointfvals


class BatchNormalizedConv1d(nn.Module):
    def __init__(self, input, output):
        super(BatchNormalizedConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(input, output, 1)    
        self.bn = nn.BatchNorm1d(output)

    def forward(self, x):
        return self.bn(self.conv(x))
    

class BatchNormalizedLinear(nn.Module):
    def __init__(self, input, output):
        super(BatchNormalizedLinear, self).__init__()
        self.fc = nn.Linear(input, output)    
        self.bn = nn.BatchNorm1d(output)

    def forward(self, x):
        return self.bn(self.fc(x))