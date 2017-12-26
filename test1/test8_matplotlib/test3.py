import matplotlib.pyplot as plt

plt.scatter(2, 4, s=200)  # s标识点的尺寸

plt.title('axis', fontsize=24)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.tick_params('both', which='major', labelsize=14)
plt.show()
