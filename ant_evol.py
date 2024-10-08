#%%

import numpy as np
import os
import matplotlib.pyplot as plt
from numba import get_num_threads, jit, njit
import lib_model
import seaborn as sns
import pandas as pd

n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 6 
numthreads = n_threads-space
print("Using", numthreads, "leaving ", space, "free")

dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

data_dir = os.path.join(proj_path,"Data","Synthetic")
if not(os.path.exists(data_dir)): os.mkdir(data_dir)
#%%
#Generate a trajectory from the data #TODO: substitute this step for actual data one working.

t_fin = 10 #final evolution time #TODO: final observation time for trajectory
dwr = 0.1 #observation time #TODO: steps between observations of the trajectory
v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
param = np.array([v,l,phi,Mu,Sigma,th0]) #"known" model parameters
beta, delta = 4,0.3
ks = np.array([beta, delta]) #This is what we want to inffer!


#Simulation setup.
h = 0.01 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
Ntraj = 150
Ntraj_ic = 1
epsilon = 0.01


#%%
#Generate data #TODO: this will be: upload trajectory.
names = ["Time","x","y","theta","id_traj"]
df = pd.DataFrame(columns=names)
for i in range(Ntraj):
    ci = np.array([0,0,np.random.uniform(np.pi/2-0.5,np.pi/2)]) #TODO: initial condition from the trajectory
    data = lib_model.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj_ic)
    xindx = np.arange(0,Ntraj_ic*len(ci),len(ci))
    yindx = np.arange(1,Ntraj_ic*len(ci),len(ci))
    thindx= np.arange(2,Ntraj_ic*len(ci),len(ci))
    for j in range(Ntraj_ic):
        df_temp = pd.DataFrame(columns=names)
        df_temp["Time"] = np.arange(0,int(Nt/iwr))*dwr
        df_temp["x"] = data[xindx[j]]
        df_temp["y"] = data[yindx[j]]
        df_temp["theta"] = data[thindx[j]]
        df_temp["id_traj"] = f"ic{i}_tr{j}"
        df = pd.concat([df,df_temp],ignore_index=True)
#%%
df.to_csv(os.path.join(data_dir,f"Synthetic-beta_{beta}-delta_{delta}-time_{t_fin}.dat"),index=False)
#%%
#Noisy points
noisy_data = data + np.random.normal(0,0.01,np.shape(data))
ts = np.arange(0,int(Nt/iwr))*dwr
# %%
#plot data
fig, ax = plt.subplots(ncols=1,nrows=4,figsize=(11,6*4))
for i in range(Ntraj):
    ax[0].plot(noisy_data[xindx[i]],noisy_data[yindx[i]],label="noisy")
    ax[0].plot(data[xindx[i]],data[yindx[i]],label="original")
    ax[0].set(xlabel="x",ylabel="y")
    ax[0].legend()
    ax[1].plot(ts,noisy_data[xindx[i]])
    ax[1].plot(ts,data[xindx[i]])
    ax[1].set(xlabel="t",ylabel="x")
    ax[2].plot(ts,noisy_data[yindx[i]])
    ax[2].plot(ts,data[yindx[i]])
    ax[2].set(xlabel="t",ylabel="y")
    ax[3].plot(ts,noisy_data[thindx[i]])
    ax[3].plot(ts,data[thindx[i]])
    ax[3].set(xlabel="t",ylabel="th")

# %%

