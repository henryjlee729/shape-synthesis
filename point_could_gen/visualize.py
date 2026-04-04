import numpy as np
import matplotlib.pyplot as plt

points = np.loadtxt("output/chair_3.pts")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='steelblue')
ax.set_title("Generated Chair")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()