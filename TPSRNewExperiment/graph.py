from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm
from matplotlib import colors
import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = 5

plt.hold(True)
n = 30
theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_zlim(-2,2)
ax1.scatter(x,y,z+0.001,color="black",s=1,marker='o')
for i in range(n):
    if(i%3==0):
        color='yellowgreen'
    elif(i%3==1):
        color="silver"
    else:
        color="cyan"
    ax1.plot_surface(x[i:i+2], y[i:i+2], z[i:i+2], rstride=3, cstride=1, edgecolors=color,color=color,alpha=0.4,shade=False)
ax1.plot_wireframe(x, y, z,rstride=3, cstride=1,color='r',alpha=0.4)
ax1.view_init(36, 26)
plt.show()
print("yes")
