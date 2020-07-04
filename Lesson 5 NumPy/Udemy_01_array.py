lst = [1, 2, 3, 4, 5]
import numpy as np

lst1 = np.array(lst)
y = lst1 / 4
print(y)
x = np.array([1, 2, 3, 4, 5])
x = x / 3.0
print(x)
x1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

x2 = np.array(x1)
print(x2)

x3 = np.arange(0, 11, 5)
print(x3)

x4 = np.zeros((2, 3))
print(x4)
x5 = np.ones((7, 4))
print(x5)

x6 = np.linspace(0, 5, 100)
print(x6)

x7 = np.eye(4)
print(x7)

x8 = np.random.rand(5, 5)
print(x8)

x9 = np.random.randn(8, 4)
print(x9)

y1 = np.random.randint(0, 100, 10)
print(y1)

y2 = np.arange(25)
print(y2)
y3 = y2.reshape(5, 5)
print(y3)

y4 = np.random.randint(0, 50, 12)
print(y4)
y5 = y4.reshape(3, 4)
print(y5)

print(y3.dtype)
