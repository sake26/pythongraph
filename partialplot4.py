import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
OPTIONS_BFGS = {'disp': False, 'gtol': 1e-20, 'xtol': 1e-20, 'eps': 1.4901161193847656e-08, 'return_all': False,  'norm': 2}

phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

fig0, ax0 = plt.subplots()
XI= 2
ARM = 0
EXP = 1
Xa=[200,20]
Xn=[200,20]
NBneutr=4
N= np.arange(40)
CostA=0.0001
CostN= 0.01
G0=500
G=12
NU=6
PRIX=[0.,0.]
ALL=0
NEUT=1
xplot0=np.array([])
xplot1=np.array([])
bound =((0,10000),(0,10000))


def target(n,x,y):
    return n*x[EXP] + NBneutr* y[EXP]
def fconv(n,x,y) :
    return x[0]*phi(target(n,x,y)/XI) - target(n,x,y) 
def FuncA(n,p,x,y):
    return n*fconv(n,x,y) - CostA*(n*(x[ARM]+x[EXP]))**2 + n*x[EXP]*p
def FuncN(n,p,x,y):
    return n*fconv(n,x,y) - n*CostN*((x[ARM]+x[EXP]))**2 + n*p*x[EXP]
def Price(p,n):
    global Xn,Xa
    FuncP = (p[ALL] * (G0 - n * Xa[EXP])) + (p[NEUT] * (G0 - NBneutr * Xn[EXP])) + (2 * NU * p[ALL] * p[NEUT]) - (
            G * (p[ALL]**2 + p[NEUT]**2))
    return -FuncP
for i in N:
    f=sc.optimize.minimize(Price, (PRIX), args=(i,), method='L-BFGS-B',options=OPTIONS_BFGS,bounds=bound).x
    print(FuncA(i,f[0],Xa,Xn))
    print(FuncN(i,f[1],Xa,Xn))
    print(f)
    xplot0=np.append(xplot0,FuncA(i,f[0],Xa,Xn))
    xplot1=np.append(xplot1,FuncN(i,f[1],Xa,Xn))
    
    
ax0.plot(N,xplot0, color='blue',label="Alliance case")
ax0.plot(N,xplot1, color='green', label="Neutral case",linestyle='dashed')
ax0.legend()
ax0.set(xlabel="Number of Nations", ylabel="Social Welfare")
plt.show()
