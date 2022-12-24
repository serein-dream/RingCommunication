import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):

  def __init__(self):
        super(CNN,self).__init__()# super用于调用父类(超类)的构造函数
        self.conv1 = nn.Sequential(
            # 二维卷积
            nn.Conv2d(in_channels=1,# 输入图片的通道数
                            out_channels=16,# 卷积产生的通道数
                            kernel_size=3,# 卷积核尺寸
                            stride=2,# 步长,默认1
                            padding=1),# 补0数，默认0
            # 数据在进入ReLU前进行归一化处理，num_features=batch_size*num_features*height*width
            # 先分别计算每个通道的方差和标准差，然后通过公式更新矩阵中每个值，即每个像素。相关参数：调整方差，调整均值
            # 输出期望输出的(N,C,W,H)中的C (数量，通道，高度，宽度)
            # 实际上，该层输入输出的shape是相同的
            nn.BatchNorm2d(16),
            # 设置该层的激活函数RELU()
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            # torch.nn.BatchNorm2d(16),
            nn.Conv2d(16,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            # torch.nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            # torch.nn.BatchNorm2d(64),
            nn.Conv2d(64,64,2,2,0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 全连接层参数设置
        self.mlp1 = nn.Linear(2*2*64,100)# 为了输出y=xA^T+b,进行线性变换（输入样本大小，输出样本大小）
        self.mlp2 = nn.Linear(100,10)
  def forward(self,x):# 前向传播
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))# 将多维度的tensor展平成一维
        x = self.mlp2(x)
        return x
  '''
  
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
        '''