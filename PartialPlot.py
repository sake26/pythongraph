import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

fig0, ax0 = plt.subplots()
XI= 1
ARM = 0
EXP = 1
Xa=[200,20]
Xn=[200,20]
NBneutr=2
N= np.arange(300)
CostA=0.0001
CostN= 0.01
Pa=100
Pn=60

y=N*Xa[EXP] + NBneutr* Xn[EXP]
fconv = Xa[0]*phi(y/XI) - y 
FuncA= N*fconv - CostA*(N*(Xa[ARM]+Xa[EXP]))**2 + N*Pa*Xa[EXP]
FuncN= N*fconv - N*CostN*((Xa[ARM]+Xa[EXP]))**2 + N*Pn*Xa[EXP]

ax0.plot(N,FuncA, color='blue',label="Alliance")
ax0.plot(N,FuncN, color='green', label="Neutral")
ax0.legend()
plt.show()
