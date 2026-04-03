import os
import tinygrad
from tinygrad import Tensor,nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph 
from tinygrad import TinyJit
from tinygrad.nn.optim import AdamW

# plot armour
data = np.load("/Users/vihantiwari/Documents/projects/gram_comp/data/1023_24-4.npz")
pos = data["pos"]
idcs_airfoil = data["idcs_airfoil"]
wing_points = pos[idcs_airfoil]
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(wing_points[:,0],wing_points[:,1],wing_points[:,2],s=0.1)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.show()
plt.close()