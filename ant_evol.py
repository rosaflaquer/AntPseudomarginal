#%%

import numpy as np
import os
import matplotlib.pyplot as plt
from numba import get_num_threads, set_num_threads
import lib_model
import seaborn as sns
import pandas as pd

n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 0
numthreads = n_threads-space
set_num_threads(numthreads)
print("Using", numthreads, "leaving ", space, "free")

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


data_dir = os.path.join(proj_path,"Data","Synthetic")
if not(os.path.exists(data_dir)): os.mkdir(data_dir)
#%%
#Generate a trajectory from the data #TODO: substitute this step for actual data one working.

t_fin = 150 
dwr = 1 
#v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
v,l,phi,Mu,Sigma,th0 = 5,13,0.9,0.0,6.5,1.0
param = np.array([v,l,phi,Mu,Sigma,th0]) #"known" model parameters
beta, delta = np.array([0.4,0.05])
ks = np.array([beta, delta]) #This is what we want to inffer!


#Simulation setup.
h = 0.01 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
Ntraj = 150
Ntraj_ic = 1



#%%

names = ["Time","x","y","theta","id_traj"]
df = pd.DataFrame(columns=names)
for i in range(Ntraj):
    ci = np.array([0,0,np.random.uniform(np.pi/2-0.5,np.pi/2)]) #TODO: initial condition from the trajectory
    #ci = np.array([0,0,np.random.uniform(3*np.pi/2-0.5,3*np.pi/2)]) #TODO: initial condition from the trajectory
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

#plot data
fig, ax = plt.subplots(ncols=1,nrows=4,figsize=(11,6*4))
for idx in df["id_traj"].unique():
    traj = df[df["id_traj"]==idx]
    ax[0].plot(traj.x,traj.y)
    ax[0].set(xlabel="x",ylabel="y")
    ax[1].plot(traj.Time,traj.x)
    ax[1].set(xlabel="t",ylabel="x")
    ax[2].plot(traj.Time,traj.y)
    ax[2].set(xlabel="t",ylabel="y")
    ax[3].plot(traj.Time,traj.theta)
    ax[3].set(xlabel="t",ylabel="th")
ax[0].set(xlim=[-60,60])
ax[1].set(ylim=[-60,60])

name = f"beta_{beta}-delta_{delta}-time_{t_fin}"
data_dir = os.path.join(proj_path,"Data","Synthetic",name)
if not(os.path.exists(data_dir)) : os.mkdir(data_dir)
fig.savefig(os.path.join(data_dir,f"Synthetic-{name}.png"),format="png",
    facecolor="w",edgecolor="w",bbox_inches="tight")
#%%

if not(os.path.exists(data_dir)): os.mkdir(data_dir)
df.to_csv(os.path.join(data_dir,f"Synthetic-{name}.dat"),index=False)
#%%
idxs = df["id_traj"].unique()[:4]
for idx in idxs:
    traj = df[df["id_traj"]==idx]
    plt.plot(traj.x,traj.y)
plt.show()

# %%

trajs = df.groupby(["id_traj"])
df["diff"] = trajs["Time"].diff(1)
df["dth"] = trajs["theta"].diff(1)/df["diff"]
df["Cl"] = lib_model.Cl(df["x"].values,df["y"].values,df["theta"].values,lib_model.phtrail,param,ks)
df["Cr"] = lib_model.Cr(df["x"].values,df["y"].values,df["theta"].values,lib_model.phtrail,param,ks)
df["Cc"] = (df["Cl"] - df["Cr"])*np.sin(df["theta"])

#%%
trajs = df.groupby(["id_traj"])
betas = v*trajs["dth"].mean()/trajs["Cc"].mean()
b_estim = np.mean(betas)
deltas = trajs["dth"].var() - (b_estim/v)**2*trajs["Cc"].var()
d_estim = np.mean(deltas)
print(b_estim)
print(d_estim)
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(12,6))
ax.axhline(beta,color="black",ls="--")
ax.plot(betas.values)
ax.axhline(b_estim,color="grey")
#ax.set(ylim=[0,0.5])
plt.show()
#%%
count = 0
for idx in df["id_traj"].unique():
    traj = df[df["id_traj"]==idx]
    count += 1
    print(traj[["theta","dth"]])
    if count > 5: break