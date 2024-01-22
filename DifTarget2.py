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
    plt.rc('axes', labelsize=13)   
    global XI
    Xeq=np.array([[ 0,0],[0. ,0.],[0., 0.]])
    
    
    global xo
    fig, (axs) = plt.subplots(1, 3)
    
        
    XI=([10.],[10],[10])
    xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
    stop=0
    t=0
    xplot0=np.array([])
    xplot1=np.array([])
    xplot2=np.array([])
    yplot0=np.array([])
    yplot1=np.array([])
    yplot2=np.array([])
        
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
                
            print(xo)  
        t +=1
    Xeq=np.copy(xo)
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
                
            print(xo)  
        xplot0=np.append(xplot0,t)
        yplot0=np.append(yplot0,((np.linalg.norm(temp[0]-Xeq[0]))))
        
        
        xplot1=np.append(xplot1,t)
        yplot1=np.append(yplot1,((np.linalg.norm(temp[1]-Xeq[1]))))
        
        xplot2=np.append(xplot2,t)
        yplot2=np.append(yplot2,((np.linalg.norm(temp[2]-Xeq[2]))))
        t +=1
    axs[0].semilogy(xplot0, yplot0, linestyle="solid",color='red',linewidth = 2,label=r'$\xi=10$', marker=',')
    axs[0].set_title("Player 1",color='red')
    
    axs[1].semilogy(xplot1, yplot1, linestyle="solid",color='red',linewidth = 2, marker=',')
    axs[1].set_title("Player 2",color='blue')
    axs[2].semilogy(xplot2, yplot2, linestyle="solid",color='red',linewidth = 2, marker=',')
    axs[2].set_title("Player 3",color='green')
    
    
    
    
    XI=([30.],[30.],[30.])
    xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
    stop=0
    t=0
    xplot0=np.array([])
    xplot1=np.array([])
    xplot2=np.array([])
    yplot0=np.array([])
    yplot1=np.array([])
    yplot2=np.array([])
        
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
                
            print(xo)  
        t +=1
    Xeq=np.copy(xo)
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
                
            print(xo)  
        xplot0=np.append(xplot0,t)
        yplot0=np.append(yplot0,((np.linalg.norm(temp[0]-Xeq[0]))))
        
        
        xplot1=np.append(xplot1,t)
        yplot1=np.append(yplot1,((np.linalg.norm(temp[1]-Xeq[1]))))
        
        xplot2=np.append(xplot2,t)
        yplot2=np.append(yplot2,((np.linalg.norm(temp[2]-Xeq[2]))))
        t +=1
    axs[0].semilogy(xplot0, yplot0, linestyle="solid",color='blue',linewidth = 2,label=r'$\xi=30$', marker=',')
    
    axs[1].semilogy(xplot1, yplot1, linestyle="solid",color='blue',linewidth = 2, marker=',')
    axs[2].semilogy(xplot2, yplot2, linestyle="solid",color='blue',linewidth = 2, marker=',')  
    
    
    
    
    XI=([50.],[50.],[50.])
    xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
    stop=0
    t=0
    xplot0=np.array([])
    xplot1=np.array([])
    xplot2=np.array([])
    yplot0=np.array([])
    yplot1=np.array([])
    yplot2=np.array([])
        
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
                
            print(xo)  
        t +=1
    Xeq=np.copy(xo)
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
                
            print(xo)  
        xplot0=np.append(xplot0,t)
        yplot0=np.append(yplot0,((np.linalg.norm(temp[0]-Xeq[0]))))
        
        
        xplot1=np.append(xplot1,t)
        yplot1=np.append(yplot1,((np.linalg.norm(temp[1]-Xeq[1]))))
        
        xplot2=np.append(xplot2,t)
        yplot2=np.append(yplot2,((np.linalg.norm(temp[2]-Xeq[2]))))
        t +=1
    axs[0].semilogy(xplot0, yplot0, linestyle="solid",color='green',label=r'$\xi=50$',linewidth = 2, marker=',')
    
    axs[1].semilogy(xplot1, yplot1, linestyle="solid",color='green',linewidth = 2, marker=',')
    axs[2].semilogy(xplot2, yplot2, linestyle="solid",color='green',linewidth = 2, marker=',')   
    
    
    
    XI=([70.],[70.],[70.])
    xo=np.array([[0.005,0.005],[0.005,0.005],[0.005,0.005]])
    stop=0
    t=0
    xplot0=np.array([])
    xplot1=np.array([])
    xplot2=np.array([])
    yplot0=np.array([])
    yplot1=np.array([])
    yplot2=np.array([])
        
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
                
            print(xo)  
        t +=1
    Xeq=np.copy(xo)
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
                
            print(xo)  
        xplot0=np.append(xplot0,t)
        yplot0=np.append(yplot0,((np.linalg.norm(temp[0]-Xeq[0]))))
        
        
        xplot1=np.append(xplot1,t)
        yplot1=np.append(yplot1,((np.linalg.norm(temp[1]-Xeq[1]))))
        
        xplot2=np.append(xplot2,t)
        yplot2=np.append(yplot2,((np.linalg.norm(temp[2]-Xeq[2]))))
        t +=1
    axs[0].semilogy(xplot0, yplot0, linestyle="solid",color='black',label=r'$\xi=70$',linewidth = 2, marker=',')
    
    axs[1].semilogy(xplot1, yplot1, linestyle="solid",color='black',linewidth = 2, marker=',')
    axs[2].semilogy(xplot2, yplot2, linestyle="solid",color='black',linewidth = 2, marker=',')
      
    for ax in axs.flat:
        ax.set(xlabel="Number of iterations", ylabel="$\|x_{n^{(k)}}-x^*_n\|$")
    fig.legend(loc='upper left', ncols=4)
    
# Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    
    plt.show()
                
                    
        
main()            