import matplotlib.pyplot as plt

x = list(range(1, 500))
y = [i ** 3 for i in x]

plt.scatter(x, y, s=20, edgecolor='none', c=y, cmap=plt.cm.Reds)
plt.axis([0, 500, 0, 125000000])
plt.show()
