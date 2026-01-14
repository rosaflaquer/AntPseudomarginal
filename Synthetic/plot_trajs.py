#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import lib_model_extended as lib

t_fin = 600
dwr = 1 
v = 6
l = 12.8
phi = 0.95
Mu = 0.0
Sigmas = [75,50]
th0 = 1.0
delta = 0.175
beta = 5
gamma = 1.2
sigma = 50
h = 0.1 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
param = np.array([v,Mu,th0,sigma,l,phi,delta]) #"known" model parameters
ks = np.array([beta,gamma]) #This is what we want to inffer!


ci = np.array([0.0, 0.0, np.pi/2*.5, 0.0, 0.0])
ci[3] = lib.Cl(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
ci[4] = lib.Cr(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
data = lib.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,1)
xindx = np.arange(0,len(ci),len(ci))
yindx = np.arange(1,len(ci),len(ci))
thindx= np.arange(2,len(ci),len(ci))

plt.plot(data[0,:], data[1,:], label='x-y trajectory')
# %%
