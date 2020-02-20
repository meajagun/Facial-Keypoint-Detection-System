## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # 2x2 maxpooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 3 fully connected hidden linear layer with batch normalization
        self.fc1 = nn.Linear(256 * 12 * 12, 1920)
        self.fc1_bn = nn.BatchNorm1d(1920)
        self.fc2 = nn.Linear(1920, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        
        # final output layer
        self.fc3 = nn.Linear(1000, 136)
        
        # dropout layer
        self.dropout = nn.Dropout(0.25) 
          
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        
        # flatten image input for fully connected layers
        x = x.view(x.size(0), -1)
        # 1st hidden layer, with dropout layer and relu activation function for batch norm and fc layer
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        # 2nd hidden layer, with dropout layer and relu activation function for batch norm and fc layer 
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        # final output layers for network
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
