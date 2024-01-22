import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm
phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

L=0.7
P=1
Costn=0.005
Costc=0.005
Alph=0.5
Xn = np.arange(0.01, 2500, 2)

Yn = np.arange(0.01, 4000, 2)
Xn, Yn  = np.meshgrid(Xn, Yn)
Xc = 0.0000376863278
Yc = 0.0000376863278



xfnuc = Xn*phi((Yn+Yc)/P) - Yn * np.exp(-(L*Xn/Yn))
yfnuc = Yn*phi((Xn+Xc)/P) - Xn * np.exp(-(L*Yn/Xn))
xfconv = Xc*phi((Yc)/P) - Yc * psi(L*Xn/Yc)
yfconv = Yc*phi((Xc)/P) - Xc * psi(L*Yn/Xc)
xfcost= 0.5*Costn*(Xn**2) +.5*Costc*(Xc**2)
yfcost= 0.5*Costn*(Yn**2) +.5*Costc*(Yc**2)

xZ = Alph *xfnuc +Alph*xfconv- xfcost 
yZ = Alph *yfnuc +Alph*yfconv- yfcost
fig, ax = plt.subplots()
CS = ax.contour(Xn,Yn,xZ, colors='k')
CS1= ax.contour(Xn,Yn,yZ)
ax.clabel(CS1, inline=True, fontsize=10)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')

plt.show()