import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1：输入3通道(RGB)，输出6通道，5x5卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化层：2x2窗口，步长2
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2：输入6通道，输出16通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1：输入16*5*5，输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2：输入120，输出84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3：输入84，输出10(对应10个类别)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积+ReLU+池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积+ReLU+池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层+ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x