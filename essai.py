import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

N=3
xo=np.array([[3.,1.],[3.,1.],[1.,4.]])
REL= np.zeros((N,N))
def AddRel(x,y):
    global REL
    REL[x,y]=1
    REL[y,x]=1
AddRel(0,1)
AddRel(1,2)
print(REL)
print(REL[1])
print(REL[1][:, None]*xo)
y = np.copy((REL[1][:, None]*xo)[np.arange(len(xo))].max(axis=0)) # Find the maxima of x^nuc, x^conv over players except 'player'.
print(y)