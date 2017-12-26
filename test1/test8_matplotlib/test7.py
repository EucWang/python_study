
import matplotlib.pyplot as plt

x = list(range(1001))
y = [i**2 for i in x]

# c设置为y值的列表
# cmap告诉pyplot使用哪个颜色渐变
# 效果就是 根据y值的变大,颜色由浅蓝变成深蓝
plt.scatter(x, y, c=y,cmap=plt.cm.Blues,edgecolor='none', s=4)

#plt.show()

# 用 savefig 方法替换show()方法,会将图片保存到文件中
plt.savefig('squares_plot.png',bbox_inches='tight')