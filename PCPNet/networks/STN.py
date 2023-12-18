import torch
from torch import nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, num_points=500, dim=3):
        super(STN, self).__init__()

        self.dim = dim
        self.num_points = num_points

        self.bnc1 = BatchNormalizedConv1d(self.dim, 64)
        self.bnc2 = BatchNormalizedConv1d(64, 128)
        self.bnc3 = BatchNormalizedConv1d(128, 1024)
        
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bnfc1 = BatchNormalizedLinear(1024, 512)
        self.bnfc2 = BatchNormalizedLinear(512, 256)

        self.fc3 = nn.Linear(256, self.dim*self.dim)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bnc3(F.relu(self.bnc2(F.relu(self.bnc1(x))))))

        # symmetric operation over all points
        x = self.mp1(x)
     
        x = x.view(-1, 1024)

        x = self.fc3(F.relu(self.bnfc2(F.relu(self.bnfc1(x)))))

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x
    

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