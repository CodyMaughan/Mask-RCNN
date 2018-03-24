import matplotlib.pyplot as plt
import numpy

fig, axes = plt.subplots(2, 1)
axes[0].scatter([1, 2, 3], [1, 2, 3])
axes[1].scatter([1, 2, 3], [3, 2, 1])
fig.suptitle('Hello World')
plt.show()