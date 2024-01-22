import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

fig0, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
XI= 1
ARM = 0
EXP = 1
Xa=[200,20]
Xn=[200,20]
NBneutr=4
N= 10
p= np.arange(0.,10,0.01)
pp= np.arange(0.,10,0.01)
P,PP = meshgrid(p,pp)
CostA=0.0001
CostN= 0.001
Pa=1
Pn=1
y=N*Xa[EXP] + NBneutr* Xn[EXP]
fconv = Xa[0]*phi(y/XI) - y 
FuncA= N*fconv - CostA*(N*(Xa[ARM]+Xa[EXP]))**2 + P*N*Pa*Xa[EXP]
FuncN= N*fconv - N*CostN*((Xa[ARM]+Xa[EXP]))**2 + PP*N*Pn*Xa[EXP]


surf = ax0.contour3D(P, PP, FuncA, 100,cmap=cm.coolwarm,
                       linewidth=0, antialiased=True,alpha=0.6)
surf2 = ax0.contour3D(P, PP, FuncN,100, cmap=cm.binary,
                       linewidth=0, antialiased=True,alpha=0.6)
fig0.colorbar(surf, shrink=0.5, aspect=5)
plt.show()