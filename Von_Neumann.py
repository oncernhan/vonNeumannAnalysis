# -*- coding: utf-8 -*-
"""
Spyder Editor

Von Newmann Stability Analysis 

Original Content: Tan Beng Hau, Cavitation Lab

Date Access: 13th Jan 2019

Author: Tan Thanh Nhan Phan 
"""


from __future__ import division 
from scipy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm #color map
import numpy as np
from math import *



from scipy import linalg #for implicit solution


def explicitDiffusion(Nt, Nx, L, T, D):
    
    """
    Performs numerical solutions for diffusion equation
    with explicit scheme
    
    Inputs:
        Nt: Number of timesteps
        Nx: Number of points in the grid
        L: length of the grid
        T: Total time
        D: constant
        
    Outputs:
        
    """
    
    dt = L/Nt
    dx = T/Nx
    alpha = D * dt / (dx**2)
    
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    u = np.zeros((Nx, Nt))
    
    """ Boundary Conditions"""
    u[0, :] = 0
    u[-1, :] = 0
    
    """Initial condition"""
    u[:, 0] = np.sin(pi*x)
    
    for j in range(Nt - 1):
        for i in range(1, Nx - 1):
            u[i, j+1] = u[i,j] + alpha*(u[i-1, j] - 2*u[i,j] + u[i+1,j])
    
    return u, x, t, alpha


def implicitDiffusion(Nt, Nx, L, T, D):
    
    """
    Performs numerical solutions for diffusion equation
    with implicit scheme
    
    Inputs:
        Nt: Number of timesteps
        Nx: Number of points in the grid
        L: length of the grid
        T: Total time
        D: constant
        
    Outputs:
        
    """
    dt = L/Nt
    dx = T/Nx
    alpha = D * dt / (dx**2)
    
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    u = np.zeros((Nx, Nt))
    
    """ Boundary Conditions"""
    u[0, :] = 0
    u[-1, :] = 0
    
    """Initial condition"""
    u[:, 0] = np.sin(pi*x)
    
    """Create a tridiagonal matrix M"""
    aa = -alpha * np.ones(Nx - 3)
    bb = (1 + 2*alpha) * np.ones(Nx - 2)
    cc = -alpha * np.ones(Nx - 3)
    M = np.diag(aa, -1) + np.diag(bb, 0) + np.diag(cc, 1)
    print(M.size)
    
    for k in range(1, Nt):
        u[1:-1, k] = np.linalg.solve(M, u[1:-1, k-1])
    
    return u, x, t, alpha 




"""Explicit Scheme Plot"""
fig = plt.figure(figsize = (14, 7))
plt.rcParams['font.size'] = 14

#CFL = 0.25
ax = fig.add_subplot(121, projection = '3d')
u, x, t, alpha = explicitDiffusion(Nt = 2500, Nx = 50, L = 1.0, T = 1.0, D = 0.25)
T, X = np.meshgrid(t, x)
N = u/u.max()
ax.plot_surface(T, X, u, linewidth = 0, facecolors = cm.jet(N), rstride = 1, cstride = 50)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = 0.25$')

#CFL = 0.505
ax = fig.add_subplot(122, projection = '3d')
u1, x1, t1, alpha1 = explicitDiffusion(Nt = 2500, Nx = 50, L = 1.0, T = 1.0, D = 0.5045)
T1, X1 = np.meshgrid(t1, x1)
N1 = u1/1.0
ax.plot_surface(T1, X1, u1, linewidth = 0, facecolors = cm.jet(N1), rstride = 1, cstride = 50)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = 0.5045$')


plt.figure(figsize = (14, 7))
plt.subplot(121)
Nt = 2500

for i in range(Nt):
    if i%300 == 0:
        plt.plot(x, u[:, i], label = i)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('$\\alpha < 0.5$')
plt.legend(loc = 'best')


plt.subplot(122)

for i in range(Nt):
    if i%300 == 0:
        plt.plot(x1, u1[:, i], label = i)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('$\\alpha > 0.5$')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

        
"""Implicit Scheme Plot"""
fig = plt.figure(figsize = (14, 7))
plt.rcParams['font.size'] = 14

#CFL = 0.25
ax = fig.add_subplot(121, projection = '3d')
u, x, t, alpha = implicitDiffusion(Nt = 2500, Nx = 50, L = 1.0, T = 1.0, D = 0.25)
T, X = np.meshgrid(t, x)
N = u/u.max()
ax.plot_surface(T, X, u, linewidth = 0, facecolors = cm.jet(N), rstride = 1, cstride = 50)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = 0.25$')

#CFL = 0.505
ax = fig.add_subplot(122, projection = '3d')
u1, x1, t1, alpha1 = implicitDiffusion(Nt = 2500, Nx = 50, L = 1.0, T = 1.0, D = 0.5045)
T1, X1 = np.meshgrid(t1, x1)
N1 = u1/u1.max()
ax.plot_surface(T1, X1, u1, linewidth = 0, facecolors = cm.jet(N1), rstride = 1, cstride = 50)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = 0.5045$')
plt.tight_layout()


plt.figure(figsize = (14, 7))
plt.subplot(121)
Nt = 2500

for i in range(Nt):
    if i%300 == 0:
        plt.plot(x, u[:, i], label = i)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('$\\alpha < 0.5$')
plt.legend(loc = 'best')


plt.subplot(122)

for i in range(Nt):
    if i%300 == 0:
        plt.plot(x1, u1[:, i], label = i)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('$\\alpha > 0.5$')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()
