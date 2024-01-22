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
plt.rc('axes', labelsize=13)   
fig0, axs = plt.subplots()
XI= 1
ARM = 0
EXP = 1
Xa=[200,20]
Xn=[200,20]
Xp=np.arange(30)
NBneutr=4
N= np.arange(35)
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
xplot0 = np.zeros((len(Xp), len(N)))
xplot1 = np.zeros((len(Xp), len(N)))



def target(n,xo):
    return n*xo + NBneutr* xo
def fconv(n,x,xo) :
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
for j, xo in enumerate(Xp):
    for i in N:
        f = sc.optimize.minimize(Price, PRIX, args=(i, xo,), method='L-BFGS-B', options=OPTIONS_BFGS, bounds=bound).x
        print(FuncA(i, f[0], Xa, xo))
        print(FuncN(i, f[1], Xa, xo))
        print(f)
        xplot0[j, i] = FuncA(i, f[0], Xa, xo)
        xplot1[j, i] = FuncN(i, f[1], Xa, xo)
print(xplot0)
surf = axs.pcolormesh(J, J1, xplot0-xplot1, linewidth=0, alpha=0.8,)
fig0.colorbar( surf,shrink=0.5, aspect=5,label='Social Welfare')
plt.title('Coalition defection')
axs.set(xlabel="Number of Nations", ylabel="Export by nation")
plt.show()

