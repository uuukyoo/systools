from net import Net
import os
from data import classes
import torch
from PIL import Image
from torchvision import transforms
PATH = './cifar_net.pth'
# 打开图片并转换为 RGB
image_dir = "./test"

# 获取文件夹下所有图片路径
image_paths = [os.path.join(image_dir, f) 
               for f in os.listdir(image_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]



# 定义与训练时相同的预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),       # CIFAR-10 输入尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# 转为 Tensor 并加 batch 维度
images = torch.stack([transform(Image.open(p).convert('RGB')) for p in image_paths])

# 加载模型
net = Net()
net.load_state_dict(torch.load(PATH))

# 使用模型进行预测
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))