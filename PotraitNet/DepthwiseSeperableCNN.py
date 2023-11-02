import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()

        hidden_dim = in_channels * expansion_factor

        # Check if we can use a residual connection
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # Pointwise convolution (1x1)
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # Depthwise separable convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # Pointwise convolution (1x1)
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # Apply residual connection if possible
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If the input and output dimensions do not match, use a projection shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x  # Store the input for the shortcut connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)  # Add the shortcut connection
        out = self.relu(out)
        return out


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self):
        super(DepthwiseSeparableCNN, self).__init__()
        # Initial convolution (224x224 -> 112x112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False), # Depthwise Convolution
            #nn.BatchNorm2d(3),
            #nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=1), # Pointwise Convolution
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Down-sampling layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32), # Depthwise Convolution
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1), # Pointwise Convolution
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64), # Depthwise Convolution
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1), # Pointwise Convolution
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Global Average Pooling and Fully Connected Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)  # Adjust 'num_classes' to your problem

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

