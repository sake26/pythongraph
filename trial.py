import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator


phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
L=0.7
P=10
Costn=0.005
Costc=0.005
Alph=0.5
Xn = np.arange(0.01, 100, 2)

Yn = np.arange(0.01, 100, 2)





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


# Plot the surface.
surf = ax.plot_surface(Xn, Yn, xZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.8)
surf2 = ax.plot_surface(Xn, Yn, yZ, cmap=cm.PiYG,
                       linewidth=0, antialiased=False,alpha=0.8)


# Customize the z axis.
ax.set_zlim(-60, 60)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.colorbar(surf2, shrink=0.5, aspect=5)

plt.show()