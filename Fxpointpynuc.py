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
xo=np.array([1.,1.,1.])
theta=1
LAMBDA= (0.7,0.7,0.7)
XI = (20,20,20)
COST = np.array([0.005,0.005,0.005])

  

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
        fnuc = x*phi(y/XI[self.player]) - y * psi(LAMBDA[self.player]*x/y)
        
        fcost = COST[self.player]*x**2
        return -(fnuc - fcost - self.theta * (np.linalg.norm(x-self.x0))**2)
    

def compute_target_levels(player):
    """ Test case : max"""
    y = np.copy(xo[np.arange(len(xo))!=player].max(axis=0)) # Find the maxima of x^nuc, x^conv over players except 'player'.
    return y
def ppm_iter(player,theta):
    J = ObjectiveFunction(player,theta)
    y = compute_target_levels(player)
    x0 = np.copy(xo[player]) #BFGS initial guess
    bound =((x0,100000))
   
    return sc.optimize.minimize(J.fun, x0, args=y, method='L-BFGS-B',options=OPTIONS_BFGS)

def main():
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
        plt.plot(t, (np.linalg.norm(xo-temp)), color='green', linestyle='solid', linewidth = 3, marker='o')
        t +=1
    plt.show()
                
                    
        
main()   