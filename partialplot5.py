import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import LinearLocator
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
OPTIONS_BFGS = {'disp': False, 'gtol': 1e-20, 'xtol': 1e-20, 'eps': 1.4901161193847656e-08, 'return_all': False,  'norm': 2}

phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

fig0, ax0 = plt.subplots()
XI= 1
ARM = 0
EXP = 1
Xa=[200,20]
Xn=[200,20]
Xp=np.arange(40)
NBneutr=4
N= np.arange(25)
CostA=0.0001
CostN= 0.01
G0=500
G=12
NU=6
PRIX=[0.,0.]
ALL=0
NEUT=1

bound =((0,10000),(0,10000))
J,J1=np.meshgrid(N,Xp)
xplot0 = np.zeros_like(J)
xplot1 = np.zeros_like(J)


def target(n,xo):
    return n*xo + NBneutr* xo
def fconv(n,x,xo:float) :
    return x[0]*phi(target(n,xo)/XI) - target(n,xo) 
def FuncA(n,p,x,xo):
    return n*fconv(n,x,xo) - CostA*(n*(x[ARM]+xo))**2 + n*xo*p
def FuncN(n,p,x,xo):
    return n*fconv(n,x,xo) - n*CostN*((x[ARM]+xo))**2 + n*p*xo
def Price(p,n,xo):
    global Xn,Xa
    FuncP = (p[ALL] * (G0 - n * xo)) + (p[NEUT] * (G0 - NBneutr * xo)) + (2 * NU * p[ALL] * p[NEUT]) - (
            G * (p[ALL]**2 + p[NEUT]**2))
    return -FuncP
for i in N:
    for xo in Xp:
        f=sc.optimize.minimize(Price, (PRIX), args=(i,xo,), method='L-BFGS-B',options=OPTIONS_BFGS,bounds=bound).x
        print(FuncA(i,f[0],Xa,xo))
        print(FuncN(i,f[1],Xa,xo))
        print(f)
        xplot0[i]=np.append(xplot0[i],FuncA(i,f[0],Xa,Xn))
        xplot1[i]=np.append(xplot1[i],FuncN(i,f[1],Xa,Xn))
    

surf = ax0.contour3D(J,J1 , xplot0,200, cmap=cm.coolwarm,
                       linewidth=0,alpha=0.6)
surf2 = ax0.contour3D(J,J1 , xplot1, 200,cmap=cm.magma,
                       linewidth=0,alpha=0.6)
ax0.legend()
ax0.set(xlabel="Number of Nations", ylabel="Utility")
plt.show()