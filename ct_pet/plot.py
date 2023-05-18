import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Change the Size of Graph using
fig = plt.figure(figsize=(1, 1))
ax = plt.axes(projection="3d")

# Create axis
axes = [1, 128, 128]

# Create Data
data = np.ones(axes)

# Control Tranperency
alpha = 0.9

# Control colour
colors = np.empty(axes + [4])

# colors[0] = [1, 0, 0, alpha]  # red
# colors[1] = [0, 1, 0, alpha]  # green
# colors[2] = [0, 0, 1, alpha]  # blue
# colors[3] = [1, 1, 0, alpha]  # yellow
# colors[4] = [1, 1, 1, alpha]  # grey

# turn off/on axis
plt.axis("off")

# Voxels is used to customizations of
# the sizes, positions and colors.
ax.voxels(data, facecolors="red", edgecolors="grey")
plt.show()
