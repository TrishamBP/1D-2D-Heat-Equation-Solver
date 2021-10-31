# 2-D Heat Equation Solver

import numpy as np
import matplotlib.pyplot as plt

L1=4.0;L2=5.0;k=.05;

def f(x,y):
    return 1*(L1/4<x)*(x<3*L1/4)*(L2/4<y)*(y<3*L2/4)


def A_mn(m,n,L1,L2):
    return (4/(L1*L2))*(2*L1*np.sin(m*np.pi/4)*np.sin(m*np.pi/2)/(np.pi*m))*(2*L2*np.sin(n*np.pi/4)*np.sin(n*np.pi/2)/(np.pi*n))
        
        
def omega_mn(m,n,L1,L2):
    return np.sqrt( (m*np.pi/L1)**2+(n*np.pi/L2)**2 )


# a_list=np.array([an(i,P) for i in n_list])
N_l=100

n_list=np.arange(N_l)+1
m_list=np.arange(N_l)+1






    # return np.sum(uu,axis=0)




x_pl=np.linspace(0,L1,200)
y_pl=np.linspace(0,L2,200)


X_pl,Y_pl=np.meshgrid(x_pl,y_pl)


x=X_pl;y=Y_pl

def u(x,y,t,N=N_l,M=N_l):
    uu=np.array([A_mn(m,n,L1,L2)*np.exp(-k*omega_mn(m,n,L1,L2)**2*t)*np.sin(np.pi*m*x/L1)*np.sin(np.pi*n*y/L2) for n in np.arange(N)+1 for m in np.arange(M)+1])
    return np.sum(uu,axis=0)

fig = plt.figure('Heat Equation')
plt.clf()

ax = fig.gca(projection='3d')


ax.plot_surface(X_pl,Y_pl,f(X_pl,Y_pl),alpha=.5)

ax.plot_surface(X_pl,Y_pl,u(X_pl,Y_pl,1.0),alpha=1.0)

# ax.plot(x_pl,u(x_pl,0.1))


ax.view_init(elev=20., azim=-40)

frames=300;
T_0=0.0;T_f=7.0;

tt=np.linspace(T_0,T_f,frames);




plt.show()    








fig2 = plt.figure('Heat Equation Animation')
plt.clf()

ax2 = fig2.gca(projection='3d')


for i in np.arange(tt.size):     
    plt.cla()
        
    ax2.plot_surface(X_pl,Y_pl,f(X_pl,Y_pl),alpha=1.0)

    ax2.plot_surface(X_pl,Y_pl,u(X_pl,Y_pl,tt[i]),alpha=.75)
    

    
    
    ax2.set_xlabel('x')
    # ax.set_xlim(pl*l1, pl*r1)
    ax2.set_ylabel('y')
    # ax.set_ylim(pl*l2, pl*r2)
    
    ax2.set_zlabel('z')
    ax2.view_init(elev=20., azim=-40)
    # plt.axis('equal');
            
    plt.savefig('FrameStore/HeatEquation2D/Heat_'+str(i).zfill(6)+'.png',format='png');
    
    print('Done Frame ' + str(i) + '/' + str(tt.size-1))
