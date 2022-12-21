import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, 3) #32,26,26
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) # 32, 13, 13
        self.conv2 = nn.Conv2d(32, 64, 3) # 64, 11, 11
        self.max_pool2 = nn.MaxPool2d(kernel_size=2) # 64, 5, 5
        self.conv3 = nn.Conv2d(64, 64, 3) # 64,3,3
        self.dnn1 = nn.Linear(64*3*3, 64) #The first full connection layer 64
        self.dnn2 = nn.Linear(64, out_features) #The first full connection layer 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        # x = x.view(128,-1)
        x = x.view(-1,64*3*3)   #The first full connection layer 64
        x = F.relu(self.dnn1(x)) #relu activate
        x = self.dnn2(x)
        return x