import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator



phi = lambda s : np.log(1+s)
psi = lambda s : np.exp(-s)

OPTIONS_BFGS = {'disp': False, 'gtol': 1e-20, 'xtol': 1e-20, 'eps': 1.4901161193847656e-08, 'return_all': False,  'norm': 2}


ALPHA = [0.5, 0.5]
N = 3
PLAYERS = np.arange(N,dtype=int)
LAMBDA = np.zeros(N)
XI = np.zeros(N)
'''COST = np.zeros((N,2))'''
xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
theta=0.001
LAMBDA= (0.8,0.8,0.8)
XI = (.001,.001,.001)
COST = np.array([[.005,0.005],[.005,0.005],[.005,00.005]])

    

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
        fcost = (.5*COST[self.player,NUC]*x[NUC]**2) + (.5*COST[self.player,CONV]*x[CONV]**2)
        return -(ALPHA[NUC] * fnuc + ALPHA[CONV] * fconv - fcost - self.theta * (np.linalg.norm(x-self.x0))**2)
    
    def jac(self,x,y):
        Jac = np.zeros(T)
        Jac[NUC] = ALPHA[NUC] * phi((y[CONV]+y[NUC])/XI[self.player]) + LAMBDA[self.player] * (ALPHA[NUC]*psi(-(LAMBDA[self.player]*x[NUC]/y[NUC])) + ALPHA[CONV]*psi(-(LAMBDA[self.player]*x[NUC]/y[CONV])))- COST[self.player,NUC]*x[NUC]
        Jac[CONV] = ALPHA[CONV] * phi(y[CONV]/XI[self.player]) - COST[self.player,CONV]*x[CONV]
        return (Jac )

def compute_target_levels1(player):
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
    y = compute_target_levels1(player)
    x0 = np.copy(xo[player]) #BFGS initial guess
    bound =((x0[0],10000),(x0[1],10000))
   
    return sc.optimize.minimize(J.fun, x0, args=y, method='L-Bfgs-B',options=OPTIONS_BFGS,bounds=bound)

def main():
    global XI
    plt.rc('axes', labelsize=13) 
    Xeq=np.array([[ 0,0],[0. ,0.],[0., 0.]])
    
    Truc = (10,20,30,40,50,60,70,80)
    bar_labels = ['10','20','30','40','50','60','70','80']
    bar_colors = ['tab:red','tab:blue','tab:green','black']
    
    global xo
    fig0, ax0 = plt.subplots()
    Value0=np.empty((0,0), float)
    Value1=np.empty((0,0), float)
    Value2=np.empty((0,0), float)
    for g in Truc:
        Value = np.empty((3),float)
        
        XI=([g],[g],[g])
        xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
        stop=0
        t=0
        
        while stop==0:
            stop=1
            temp=np.copy(xo)
            
            for i in PLAYERS:
                
                f= ppm_iter(i ,1.1)
                
                print(f.x)
                ff=np.copy(f.x)
                print(ff)
                
                
                
                if stop==1:
                    if np.array_equal(ff,temp[i]):
                        stop=1
                    else:
                        stop=0
                xo[i]=np.copy(f.x)
                Value[i]=np.copy(f.fun)
                
                print(xo)  
            t +=1
        if (g==1):
            xg=Value
            
        Value0=np.append(Value0,Value[0])
        Value1=np.append(Value1,Value[1])
        Value2=np.append(Value2,Value[2])
    species = ('10','20','30','40','50','60','70','80')
    penguin_means = {
    'Player 1': Value0,
    'Player 2': Value1,
    'Player 3': Value2,
    }
    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    plt.rc('axes', labelsize=30) 
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        ax0.bar(x + offset, measurement+700, width, label=attribute,bottom=-700)
        multiplier += 1
    ax0.set(xlabel=r'$\xi$' , ylabel=r"$J^{*}_n$")
    ax0.set_xticks(x + width, species)
    ax0.legend(loc='upper left', ncols=3)
    
    
    print("la valeur de 0 : ",Value0)
    print("la valeur de 1 : ",Value1)
    print("la valeur de 2 : ",Value2)
    print("valeur", Value)
    plt.show()
                
                    
        
main()            