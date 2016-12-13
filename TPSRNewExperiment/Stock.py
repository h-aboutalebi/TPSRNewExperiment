# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import *
# from environment import *
# from states import *
# import matplotlib.pyplot as plt
# import numpy as np
# from random import random, seed
# from matplotlib import cm
# from matplotlib import colors
# import matplotlib as mpl
#
# class ConstructEnvironment:
#
# #BiasIndex determine the distance between each two column in the torus. n determines the number of nodes.(The true number of nodes is n**2)
#     def __init__(self,BiasIndex=3,n=30):
#         self.create(BiasIndex,n)
#
#
#     def create(self,BiasIndex,nodes):
#         self.env=environment([0,1,2,3],[0,1,2])
#         plt.hold(True)
#         n = nodes
#         SpaceColumn=BiasIndex
#         theta = np.linspace(0, 2. * np.pi, n)
#         phi = np.linspace(0, 2. * np.pi, n)
#         theta, phi = np.meshgrid(theta, phi)
#         c, a = 2, 1
#         x = (c + a * np.cos(theta)) * np.cos(phi)
#         y = (c + a * np.cos(theta)) * np.sin(phi)
#         z = a * np.sin(theta)
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111, projection='3d')
#         ax1.view_init(36, 26)
#         ax1.set_zlim(-2, 2)
#         ax1.scatter(x, y, z + 0.001, color="w", s=4, marker='o')
#         for i in range(n):
#             if (i % SpaceColumn == 0):
#                 color = 'yellowgreen'
#             elif (i % SpaceColumn == 1):
#                 color = "silver"
#             else:
#                 color = "cyan"
#             ax1.plot_surface(x[i:i + 2], y[i:i + 2], z[i:i + 2], rstride=1, cstride=1, edgecolors=color, color=color,alpha=1, shade=True, linewidth=0)
#         StatesList=[]
#         for j in range(n):
#             Flag=False
#             Actions =[1/2,1/2]
#             if(n%SpaceColumn==0):
#                 Flag=True
#                 Actions=[1/4,1/4,1/4,1/4]
#             for i in range(n):
#                 if(i!=0):
#                     state=states(Actions,)
#                     StatesList[j-1][i]
#         ax1.plot_wireframe(x, y, z, rstride=SpaceColumn, cstride=1, color='r', alpha=0.4, linewidth=2)
#         plt.show()
#         print("yes")


# ...........................................................

def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

print(baseN(8,3))