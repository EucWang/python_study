
import matplotlib.pyplot as plt

x = list(range(1,1001))

y = [i**2 for i in x]

plt.scatter(x, y, s=1)


# 指定坐标轴的取值范围, 
plt.axis([0,1100,0,1100000])

plt.show()