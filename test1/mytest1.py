
import numpy as np
import matplotlib.pyplot as plt

dataSet = [
    [-0.017612, 14.053064],
    [-1.395634, 4.662541],
    [-0.752157, 6.538620],
    [-1.322371, 7.152853],
    [0.423363, 11.054677],
    [0.406704, 7.067335],
    [0.667394, 12.741452],
    [-2.460150, 6.866805],
    [0.569411, 9.548755],
    [-0.026632, 10.427743],
    [0.850433, 6.920334],
    [1.347183, 13.175500],
    [1.176813, 3.167020],
    [-1.781871, 9.097953]
]

# np.mat()  生成一个Matrix对象
#  .T   对这个对象进行转置
dataMat = np.mat(dataSet).T

# 将转置之后的矩阵 生成 两个List,
plt.scatter(dataMat[0].tolist(), dataMat[1].tolist(), c='red', marker='o')

x = np.linspace(-2, 2, 10)   # 返回-2 到2 之间的100个数
y = 2.8 * x + 9     #
print(x)

plt.plot(x, y)

plt.show()