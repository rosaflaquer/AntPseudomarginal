#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numba import get_num_threads
from lib_MCMC import *


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

# %%
#1) Load observations
data_dir = os.path.join(proj_path,"Data","Synthetic")
data_file = os.listdir(data_dir)[0]
data_file= "Synthetic-beta_4-delta_0.3-time_10.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))
id_traj = datadf["id_traj"].unique()
Ntrajs = len(id_traj)
len_trajs = len(datadf[datadf["id_traj"]==id_traj[0]])
data = np.zeros((Ntrajs,4,len_trajs))
sdat = 0.005
for i,id in enumerate(id_traj):
    day_traj = datadf[datadf["id_traj"]==id]
    data[i][0] = day_traj["x"].values     + np.random.normal(0,sdat,size=len(day_traj["x"]))
    data[i][1] = day_traj["y"].values     + np.random.normal(0,sdat,size=len(day_traj["y"]))
    data[i][2] = day_traj["theta"].values + np.random.normal(0,sdat,size=len(day_traj["theta"]))
    data[i][3] = day_traj["Time"].values

plt.plot(data[1,0,:],data[1,1,:])
# %%
out_path = os.path.join(proj_path,"Data","Simulations")
df = pd.read_csv(os.path.join(out_path,"Chains.dat"))
nparam = 2
C = int(len(df.columns)/2)
M = len(df)
chains = []
for i in range(C):
    chains.append([df[f"beta_{i}"].values,df[f"delta_{i}"].values])
zetas,hatR,W,V,Se = convergence(chains,nparam,C,M)
print(hatR,Se)
#%%
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
fig1,ax1=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.set(ylabel=r"$\beta$",xlabel="MC step")
ax1.set(ylabel=r"$\delta$",xlabel="MC step")
betas = np.zeros(M*C)
deltas = np.zeros(M*C)
for i in range(C):
    ax.plot(chains[i][0])
    ax1.plot(chains[i][1])
    betas[i*M:(i+1)*M] = chains[i][0]
    deltas[i*M:(i+1)*M]= chains[i][1]
plt.show()
#%%
nbins = 25
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.hist(betas,bins=nbins,density=True)
ax.axvline(np.mean(betas),color="black")
ax.axvline(np.mean(betas)+np.std(betas),color="black")
ax.axvline(np.mean(betas)-np.std(betas),color="black")
plt.show()
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.hist(deltas,bins=nbins,density=True)
ax.axvline(np.mean(deltas),color="black")
ax.axvline(np.mean(deltas)+np.std(deltas),color="black")
ax.axvline(np.mean(deltas)-np.std(deltas),color="black")
plt.show()