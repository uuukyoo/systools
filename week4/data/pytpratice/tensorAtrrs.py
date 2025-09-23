import torch

# 创建一个示例张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, 
                 requires_grad=True, device=device)

print(x)

# 打印常见属性
print("\n--- ATTRIBUTES ---")

print("Shape:", x.shape)  # 获取形状
print("Size:", x.size())  # 获取尺寸
print("Data Type:", x.dtype)  # 数据类型
print("Device:", x.device)  # 设备
print("Dimensions:", x.dim())  # 维度数
print("Total Elements:", x.numel())  # 元素总数
print("Requires Grad:", x.requires_grad)  # 是否启用梯度
print("Is CUDA:", x.is_cuda)  # 是否在 GPU 上
print("Is Contiguous:", x.is_contiguous())  # 是否连续存储