import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from data import *
from net import Net
def main():
    # 实例化网络
    net = Net()
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # 训练10个epoch
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            
            # 打印统计信息
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个mini-batch打印一次
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total:.2f}%')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')

    # 保存模型参数
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
if __name__ == "__main__":
    main()