import torch
import torch.nn as nn
import numpy as np
from DepthwiseSeperableCNN import *

class PotraitNet(nn.Module):
    def __init__(self, channelRatio=1.0, minChannel=16, weightInit=True):
        super(PotraitNet, self).__init__()

        self.minChannel = 16
        self.n_class = 2
        self.channelRatio = channelRatio

        self.stage0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.depth(32), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=self.depth(32), eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(inplace=True)
            ) 
        
        self.stage1 = self.multipleInvertedResidualBlocks([[32, 16, 1, 1]])
        self.stage2 = self.multipleInvertedResidualBlocks([[16, 24, 2, 6], [24, 24, 1, 6]])
        self.stage3 = self.multipleInvertedResidualBlocks([[24, 32, 2, 6], [32, 32, 1, 6], [32, 32, 1, 6]])
        self.stage4 = self.multipleInvertedResidualBlocks([[32, 64, 2, 6], [64, 64, 1, 6], [64, 64, 1, 6], [64, 64, 1, 6]])
        self.stage5 = self.multipleInvertedResidualBlocks([[64, 96, 1, 6], [96, 96, 1, 6], [96, 96, 1, 6]])
        self.stage6 = self.multipleInvertedResidualBlocks([[96, 160, 2, 6], [160, 160, 1, 6], [160, 160, 1, 6]])
        self.stage7 = self.multipleInvertedResidualBlocks([[160, 320, 1, 6]])

        self.deconv1 = nn.ConvTranspose2d(self.depth(96), self.depth(96), 
                                            groups=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(self.depth(32), self.depth(32), 
                                            groups=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(self.depth(24), self.depth(24), 
                                            groups=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(self.depth(16), self.depth(16), 
                                            groups=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(self.depth(8),  self.depth(8),  
                                            groups=1, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.transit1 = ResidualBlock(self.depth(320), self.depth(96))
        self.transit2 = ResidualBlock(self.depth(96),  self.depth(32))
        self.transit3 = ResidualBlock(self.depth(32),  self.depth(24))
        self.transit4 = ResidualBlock(self.depth(24),  self.depth(16))
        self.transit5 = ResidualBlock(self.depth(16),  self.depth(8))

        self.pred = nn.Conv2d(self.depth(8), self.n_class, 3, 1, 1, bias=False)

        self.edge = nn.Conv2d(self.depth(8), self.n_class, 3, 1, 1, bias=False)
        
    def forward(self, x):
        feature_1_4  = self.stage2(self.stage1(self.stage0(x)))
        feature_1_8  = self.stage3(feature_1_4)
        feature_1_16 = self.stage5(self.stage4(feature_1_8))
        feature_1_32 = self.stage7(self.stage6(feature_1_16))
        
        up_1_16 = self.deconv1(self.transit1(feature_1_32))
        up_1_8  = self.deconv2(self.transit2(feature_1_16 + up_1_16))
        up_1_4  = self.deconv3(self.transit3(feature_1_8 + up_1_8))
        up_1_2  = self.deconv4(self.transit4(feature_1_4 + up_1_4))
        up_1_1  = self.deconv5(self.transit5(up_1_2))
        
        pred = self.pred(up_1_1)
        edge = self.edge(up_1_1)
        return pred, edge


    def multipleInvertedResidualBlocks(self, inputs):
        """
        Add as many Inverted Residual Blocks as specified by the lenght od the array of the parameters
        (each block should contain a value for the input, output, stride, and expand ration)
        """
        layers = []
        for input, output, stride, expand_ratio in inputs:
            layers.append(InvertedResidual(input, output, stride, expand_ratio))
        
        return nn.Sequential(*layers)


    def depth(self, channels):
        return np.clip(channels, self.minChannel, int(channels*self.channelRatio))