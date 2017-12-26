import matplotlib.pyplot as plt

x = list(range(1,1001))
y = [i** 2 for i in x]

#plt.scatter(x,y, edgecolor='none', c='red', s=40)
#plt.scatter(x,y,s=40)
plt.scatter(x,y, edgecolor='none', c=(0,0,0.8), s=40)

plt.show()