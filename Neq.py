import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

OPTIONS_BFGS = {'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None, 'norm': 2}


ALPHA = [0.5, 0.5]
N = 3
PLAYERS = np.arange(N,dtype=int)
LAMBDA = np.zeros(N)
XI = np.zeros(N)
'''COST = np.zeros((N,2))'''
xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
theta=1
LAMBDA= (0.7,0.7,0.7)
XI = (200,200,200)
COST = np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])

    

NUC = 0
CONV = 1
TYPES = [NUC, CONV]
T = len(TYPES)

class ObjectiveFunction:
    def __init__(self,p,t):
        self.player = p
        self.theta = t
        self.x0 = xo[p]
        return
    
    def fun(self,x,y):
        fnuc = x[NUC]*phi((y[CONV]+y[NUC])/XI[self.player]) - y[NUC] * psi(LAMBDA[self.player]*x[NUC]/y[NUC])
        fconv = x[CONV]*phi(y[CONV]/XI[self.player]) - y[CONV] * psi(LAMBDA[self.player]*x[NUC]/y[CONV])
        fcost = .5*COST[self.player,NUC]*x[NUC]**2 + .5*COST[self.player,CONV]*x[CONV]**2
        return -(ALPHA[NUC] * fnuc + ALPHA[CONV] * fconv - fcost - self.theta * (np.linalg.norm(x-self.x0))**2)
    
    def jac(self,x,y):
        Jac = np.zeros(T)
        Jac[NUC] = ALPHA[NUC] * phi((y[CONV]+y[NUC])/XI[self.player]) + LAMBDA[self.player] * (ALPHA[NUC]*psi(-(LAMBDA[self.player]*x[NUC]/y[NUC])) + ALPHA[CONV]*psi(-(LAMBDA[self.player]*x[NUC]/y[CONV])))- COST[self.player,NUC]*x[NUC]
        Jac[CONV] = ALPHA[CONV] * phi(y[CONV]/XI[self.player]) - COST[self.player,CONV]*x[CONV]
        return (Jac )

def compute_target_levels(player):
    """ Test case : max
    """
    y = xo[np.arange(len(xo))!=player].max(axis=0) # Find the maxima of x^nuc, x^conv over players except 'player'.
    return y
def ppm_iter(player,theta):
    J = ObjectiveFunction(player,theta)
    y = compute_target_levels(player)
    x0 = xo[player] #BFGS initial guess
    bound =sc.optimize.Bounds([x0[NUC],x0[CONV]],[10000,10000])
   
    return sc.optimize.minimize(J.fun, x0, args=y, method='Nelder-Mead',options=OPTIONS_BFGS)

def main():
    global xo
    stop=0
    while stop==0:
        stop=1
        
        for i in PLAYERS:
            
            print(xo[i])
            f= ppm_iter(i,0.5)
            
            
            print(f.x)
            
            if stop==1:
                if np.array_equal(f.x,xo[i]):
                    stop=1
                else:
                    stop=0
            xo[i]=f.x
                
                    
        
main()            
                
                
    