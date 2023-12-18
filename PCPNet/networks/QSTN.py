import torch
from torch import nn
import torch.nn.functional as F

class QSTN(nn.Module):
    def __init__(self, num_points=500, dim=3):
        super(QSTN, self).__init__()

        self.dim = dim
        self.num_points = num_points
        
        self.bnc1 = BatchNormalizedConv1d(self.dim, 64)
        self.bnc2 = BatchNormalizedConv1d(64, 128)
        self.bnc3 = BatchNormalizedConv1d(128, 1024)

        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bnfc1 = BatchNormalizedLinear(1024, 512)
        self.bnfc2 = BatchNormalizedLinear(512, 256)
        self.fc3 = nn.Linear(256, 4)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bnc3(F.relu(self.bnc2(F.relu(self.bnc1(x))))))

        # symmetric operation over all points
        x = self.mp1(x)

        x = x.view(-1, 1024)

        x = self.fc3(F.relu(self.bnfc2(F.relu(self.bnfc1(x)))))

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = self.batch_quat_to_rotmat(x)

        return x
    
    # quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
    def batch_quat_to_rotmat(self, q, out=None):

        batchsize = q.size(0)

        if out is None:
            out = q.new_empty(batchsize, 3, 3)

        # 2 / squared quaternion 2-norm
        s = 2/torch.sum(q.pow(2), 1)

        # coefficients of the Hamilton product of the quaternion with itself
        h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

        out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
        out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
        out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

        out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
        out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
        out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

        out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
        out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
        out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

        return out

    def cos_angle(v1, v2):
        return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


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