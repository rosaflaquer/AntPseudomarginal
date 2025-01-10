#%%
import numpy as np
from numba import njit, prange

@njit
def phtrail(x,y,known_param,estimate_param):
    v,Mu,th0,Sigma,l,phi,gamma = known_param
    beta, delta = estimate_param
    return np.exp(-((x-Mu)/Sigma)**2)

@njit
def Cl(x,y,theta,trail,known_param,estimate_param):
    v,Mu,th0,Sigma,l,phi,gamma = known_param
    beta, delta = estimate_param
    return trail(x + l*np.cos(th0*(theta + phi)),y + l*np.sin(th0*(theta + phi)),known_param,estimate_param)

@njit
def Cr(x,y,theta,trail,known_param,estimate_param):
    v,Mu,th0,Sigma,l,phi,gamma = known_param
    beta, delta = estimate_param
    return trail(x + l*np.cos(th0*(theta - phi)),y + l*np.sin(th0*(theta - phi)),known_param,estimate_param)

@njit
def qs(zz,t,known_param,estimate_param):
    v,Mu,th0,Sigma,l,phi,gamma = known_param
    beta, delta = estimate_param
    x,y,th,al,ar = zz[0],zz[1],zz[2],zz[3],zz[4]
    q1 = v*np.cos(th0*(th))
    q2 = v*np.sin(th0*(th))
    q3 = beta/v*(al-ar)*np.abs(np.sin(th0*th))
    q4 = gamma*(Cl(x,y,th,phtrail,known_param,estimate_param)-al)
    q5 = gamma*(Cr(x,y,th,phtrail,known_param,estimate_param)-ar)
    return np.array([q1,q2,q3,q4,q5])

@njit
def gs(zz,t,known_param,estimate_param):
    v,Mu,th0,Sigma,l,phi,gamma = known_param
    beta, delta = estimate_param
    g1 = 0
    g2 = 0
    g3 = delta
    g4 = 0
    g5 = 0
    return np.array([g1,g2,g3,g4,g5])

@njit
def ws(sqh):
    w1 = 0
    w2 = 0
    w3 = sqh*np.random.normal(0,1)
    w4 = 0
    w5 = 0
    return np.array([w1,w2,w3,w4,w5])

@njit
def step_addMilstein(zz,t,h,sqh,known_param,estimate_param):
    w = ws(sqh)
    return zz + h*qs(zz,t,known_param,estimate_param) + gs(zz,t,known_param,estimate_param)*w

@njit
def timeev(zz,h,sqh,Nt,iwr,known_param,estimate_param):
    Neq = len(zz)
    traj = np.zeros((int(Nt/iwr),Neq))
    t = 0
    traj[0] = zz
    for i in range(1,Nt):
        t = i*h
        zz = step_addMilstein(zz,t,h,sqh,known_param,estimate_param)
        zz[2] = zz[2]%(2*np.pi)
        if i%iwr == 0: traj[i//iwr] = zz
    return traj

@njit(parallel =True)
def multiple_traj(zz,h,sqh,Nt,iwr,known_param,estimate_param,Ntraj):
    Neq = len(zz)
    data = np.zeros((Neq*Ntraj,int(Nt/iwr)))
    for i in prange(Ntraj):
        timeevol = timeev(zz,h,sqh,Nt,iwr,known_param,estimate_param)
        for j in range(Neq):
            data[Neq*i+j] = timeevol[:,j]
    return data

@njit
def model_step(zz,t0,h,sqh,Nt,known_param,estimate_param):
    t = t0
    for i in range(Nt):
        t = t0 + (i+1)*h
        zz = step_addMilstein(zz,t,h,sqh,known_param,estimate_param)
        zz[2] = zz[2]%(2*np.pi)
    return zz,t