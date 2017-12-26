
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,4,9,16,25]

plt.scatter(x, y,s=10)

plt.title('Square Number', fontsize=24)
plt.xlabel("Number", fontsize=14)
plt.ylabel("Square", fontsize=14)
#plt.tick_params('both', which='major', labelsize=14)
plt.show()