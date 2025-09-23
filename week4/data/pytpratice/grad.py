import torch
# 创建一个需要计算梯度的张量
x = torch.randn(2, 2, requires_grad=True)
print(x)

# 执行某些操作

y = x + 2
z = y * y * 3
out = z.mean()

print(out)

# 反向传播，计算梯度
out.backward()

# 查看 x 的梯度
print(x.grad)