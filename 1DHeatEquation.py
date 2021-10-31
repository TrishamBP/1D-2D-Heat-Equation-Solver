#1-D Heat Equation Solver

import numpy as np
import matplotlib.pyplot as plt

L=1.0;k=.05;

def f(x):
    return x*(0<x)*(x<L/2)+(L-x)*(L/2<x)*(x<L)


def bn(n,L):
    I1=L**2/(np.pi*n)**2*(np.sin(np.pi*n/2)-np.pi*n*np.cos(np.pi*n/2)/2)
    I2=L**2/(np.pi*n)**2*(np.sin(np.pi*n/2)+np.pi*n*np.cos(np.pi*n/2)/2)
    return 2/L*(I1+I2)


# a_list=np.array([an(i,P) for i in n_list])
N_l=200

n_list=np.arange(N_l)+1

b_n=np.array([bn(n,L) for n in n_list])
omega_n=np.array([n*np.pi/L for n in n_list])


def u(x,t,N=N_l):
    uu=np.array([b_n[i]*np.exp(-k*omega_n[i]**2*t)*np.sin(np.pi*n_list[i]*x/L) for i in np.arange(N)])
    return np.sum(uu,axis=0)




x_pl=np.linspace(0,L,1000)

fig = plt.figure('Heat Equation')
plt.clf()

ax = fig.gca()


ax.plot(x_pl,f(x_pl))

ax.plot(x_pl,u(x_pl,0.1))

ax.plot


frames=300;
T_0=0.0;T_f=7.0;

tt=np.linspace(T_0,T_f,frames);




plt.show()    








fig2 = plt.figure('Heat Equation Animation')
plt.clf()

ax2 = fig2.gca()


for i in np.arange(tt.size):     
    plt.cla()
        
    ax2.plot(x_pl,f(x_pl))

    ax2.plot(x_pl,u(x_pl,tt[i]))
    

    
    
    ax2.set_xlabel('x')
    # ax.set_xlim(pl*l1, pl*r1)
    ax2.set_ylabel('T')
    # ax.set_ylim(pl*l2, pl*r2)
    
    plt.axis('equal');
            
    plt.savefig('FrameStore/HeatEquation1D/Heat_'+str(i).zfill(6)+'.png',format='png');
    
    print('Done Frame ' + str(i) + '/' + str(tt.size-1))
