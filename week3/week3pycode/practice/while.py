n = int(input("输入一个正整数"))
i = 1
result = 1
while i <= n:
    result *= i
    i += 1
print(f"{n}的阶乘等于{result}")