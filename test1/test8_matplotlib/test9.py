import matplotlib.pyplot as plt
import numpy as np

x = [0.01, 0.05,
     0.1, 0.15,
     0.2, 0.25,
     0.3, 0.35,
     0.4, 0.45,
     0.5, 0.55,
     0.6, 0.65,
     0.7, 0.75,
     0.8, 0.85,
     0.90, 0.95
     ]
y = [0.5 * np.log((1-i)/i) for i in x]
# plt.scatter(x, y, s=20, edgecolor='none', c=y, cmap=plt.cm.Reds)
plt.plot(x, y)
plt.grid(True)
# plt.axis([0, 500, 0, 125000000])
plt.show()
