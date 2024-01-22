import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

OPTIONS_BFGS = {'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'norm': 2}


ALPHA = [1]
N = 3
PLAYERS = np.arange(N,dtype=int)
LAMBDA = np.zeros(N)
XI = np.zeros(N)
'''COST = np.zeros((N,2))'''
xo=np.array([0.005,0.005,00.005])
theta=1
LAMBDA= (0.8,0.4,0.8)
XI = (2,2,2)
COST = np.array([5.,5.00,5.00])

  

NUC = 0
TYPES=[NUC] 
T = len(TYPES)

class ObjectiveFunction:
    def __init__(self,p,t):
        self.player = p
        self.theta = t
        self.x0 = xo[p]
        return
    
    def fun(self,x,y):
        fnuc = x*phi(y/XI[self.player]) 
        
        fcost = .5*COST[self.player]*x**2
        return -.5*(fnuc - fcost - self.theta * (np.linalg.norm(x-self.x0))**2)
    

def compute_target_levels(player):
    if player==0:
        """ Test case : max"""
        y = np.copy(xo[np.arange(len(xo))!=player].max(axis=0)) # Find the maxima of x^nuc, x^conv over players except 'player'.
    if player==1:
        y = np.copy(xo[np.arange(len(xo))!=player].sum(axis=0))
    if player==2:
        y = np.copy(xo[np.arange(len(xo))!=player].mean(axis=0))
    print(player,y)
    return y
def ppm_iter(player,theta):
    J = ObjectiveFunction(player,theta)
    y = compute_target_levels(player)
    x0 = np.copy(xo[player]) #BFGS initial guess
    bound =([x0,100000])
   
    return sc.optimize.minimize(J.fun, x0, args=y, method='L-BFGS-B',options=OPTIONS_BFGS)

def main():
    Value = np.array([[0.005],[0.005],[000.005]])
    global xo
    stop=0
    t=0
    while stop==0:
        stop=1
        temp=np.copy(xo)
        
        for i in PLAYERS:
            
            f= ppm_iter(i,0.1)
            
            print(f.x)
            ff=np.copy(f.x)
            print(ff)
            print(temp)
            
            if stop==1:
                if ff==temp[i]:
                    stop=1
                else:
                    stop=0
            xo[i]=np.copy(f.x)
            Value[i]=f.fun
        plt.semilogy(t, (np.linalg.norm(xo-temp)), color='green', linestyle='solid', linewidth = 3, marker='o')
        t +=1
    print("value : ",Value)
    plt.show()
                
                    
        
main()   