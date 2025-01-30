#%%

import numpy as np
import os
import matplotlib.pyplot as plt
from numba import get_num_threads, set_num_threads
import lib_model_extended as lib
import pandas as pd

#%%

numthreads = 12
set_num_threads(numthreads)
print("Using", numthreads)

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


data_dir = os.path.join(proj_path,"Data","Synthetic","Delay")
if not(os.path.exists(data_dir)): os.mkdir(data_dir)
#%%
#Generate a trajectory from the data #TODO: substitute this step for actual data one working.

t_fin = 150 
dwr = 1 
#v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
v,l,phi,Mu,Sigma,th0,gamma = 5,13,0.9,0.0,15.0,1.0,0.07
param = np.array([v,Mu,th0,Sigma,l,phi]) #"known" model parameters
param = np.array([v,Mu,th0,Sigma,l,phi,gamma]) #"known" model parameters
beta, delta= np.array([0.4,0.075]) #,0.7])
ks = np.array([beta, delta,gamma]) #This is what we want to inffer!
ks = np.array([beta, delta]) #This is what we want to inffer!


#Simulation setup.
h = 0.01 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
Ntraj = 150
Ntraj_ic = 1

#%%

names = ["Time","x","y","theta","dif","vx","vy","v","id_traj"]
df = pd.DataFrame(columns=names)
for i in range(Ntraj):
    #ci = np.array([0,0,np.random.uniform(np.pi/2-0.5,np.pi/2),0,0]) #TODO: initial condition from the trajectory
    ci = np.array([0,0,np.pi/2-0.2,0,0]) #TODO: initial condition from the trajectory
    ci[3] = lib.Cl(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
    ci[4] = lib.Cr(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
    #ci = np.array([0,0,np.random.uniform(3*np.pi/2-0.5,3*np.pi/2)]) #TODO: initial condition from the trajectory
    data = lib.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj_ic)
    xindx = np.arange(0,Ntraj_ic*len(ci),len(ci))
    yindx = np.arange(1,Ntraj_ic*len(ci),len(ci))
    thindx= np.arange(2,Ntraj_ic*len(ci),len(ci))
    for j in range(Ntraj_ic):
        df_temp = pd.DataFrame(columns=names)
        df_temp["Time"] = np.arange(0,int(Nt/iwr))*dwr
        df_temp["x"] = data[xindx[j]]
        df_temp["y"] = data[yindx[j]]
        df_temp["theta"] = data[thindx[j]]
        df_temp["dif"] = df_temp["Time"].diff()
        df_temp["vx"] = (df_temp["x"].shift(-1) - df_temp["x"])/df_temp["dif"]
        df_temp["vy"] = (df_temp["y"].shift(-1) - df_temp["y"])/df_temp["dif"]
        df_temp["v"]  = np.sqrt(df_temp["vx"].pow(2)+df_temp["vy"].pow(2)) 
        df_temp["id_traj"] = f"ic{i}_tr{j}"
        df = pd.concat([df,df_temp],ignore_index=True)
        del(df_temp)

#%%

ths = []
xs = []
ys = []
DT = 1
for idx in df["id_traj"].unique()[:500]:
    traj = df[df["id_traj"]==idx]
    ths.append(traj.theta.values[DT])
    xs.append(traj.x.values[DT])
    ys.append(traj.y.values[DT])
#%%
#plot histos
nb = 50
ths = np.array(ths)
plt.hist((ths-ths.mean())*180/np.pi,bins=nb)
plt.title("Theta distribution")
plt.show()
xs = np.array(xs)
plt.hist(xs-xs.mean(),bins=nb)
plt.title("x distribution")
plt.show()
ys = np.array(ys)
plt.hist(ys-ys.mean(),bins=nb)
plt.title("y distribution")
plt.show()
#%%
#plot autocorrelation

fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(9,5))
traj_ids = df["id_traj"].unique()
num_trajs =len(traj_ids)
colors_lines = plt.cm.Greys(np.linspace(0.1,0.9,num_trajs))
for i,id in enumerate(traj_ids):
    traj = df[df["id_traj"] == id].copy()
    traj =traj.dropna()
    auto_corr = np.correlate(traj["vy"] - traj["vy"].mean(), traj["vy"] - traj["vy"].mean(), mode='full')[len(traj) - 1:] # Only take second half
    auto_corr /= auto_corr[0]  # Normalize
    ax.plot(auto_corr,color=colors_lines[i])
ax.set(xlabel=r"$t$", ylabel=r"$\langle \rho_y(0)\rho_y(t) \rangle/\langle \rho_y(0)\rho_y(0) \rangle$",
       title=r"$\rho_y$")
plt.show()
#fig.savefig(os.path.join(out_path,"Autocorr_rhoy.png"),format="png",
#        facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
#plot autocorrelation

fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(9,5))
traj_ids = df["id_traj"].unique()
num_trajs =len(traj_ids)
colors_lines = plt.cm.Greys(np.linspace(0.3,0.9,num_trajs))
colors_lines_in = plt.cm.YlOrRd(np.linspace(0.1,0.9,num_trajs))
for i,id in enumerate(traj_ids):
    traj = df[df["id_traj"] == id].copy()
    traj =traj.dropna()
    auto_corr = np.correlate(traj["v"] - traj["v"].mean(), traj["v"] - traj["v"].mean(), mode='full')[len(traj) - 1:] # Only take second half
    auto_corr /= auto_corr[0]  # Normalize
    if np.any(traj["x"].abs()>50):  
        color = colors_lines[i]
        alpha = 0.15
    else: 
        color = colors_lines_in[i]
        alpha = 1
    ax.plot(auto_corr,color=color,alpha=alpha)
ax.set(xlabel=r"$t$", ylabel=r"$\langle \rho(0)\rho(t) \rangle/\langle \rho(0)\rho(0) \rangle$",
       title=r"$\rho$")
plt.show()
#fig.savefig(os.path.join(out_path,"Autocorr_rhoy.png"),format="png",
#        facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
#plot autocorrelation

fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(9,5))
traj_ids = df["id_traj"].unique()
num_trajs =len(traj_ids)
colors_lines = plt.cm.Greys(np.linspace(0.1,0.9,num_trajs))
colors_lines_in = plt.cm.YlOrRd_r(np.linspace(0.1,0.9,num_trajs))
for i,id in enumerate(traj_ids):
    traj = df[df["id_traj"] == id].copy()
    traj =traj.dropna()
    auto_corr = np.correlate(traj["vx"] - traj["vx"].mean(), traj["vx"] - traj["vx"].mean(), mode='full')[len(traj) - 1:] # Only take second half
    auto_corr /= auto_corr[0]  # Normalize
    if np.any(traj["x"].abs()>50):  
        color = colors_lines[i]
        alpha = 0.15
    else: 
        color = colors_lines_in[i]
        alpha = 0.9
    ax.plot(auto_corr,color=color,alpha=alpha)
ax.set(xlabel=r"$t$", ylabel=r"$\langle \rho_x(0)\rho_x(t) \rangle/\langle \rho_x(0)\rho_x(0) \rangle$",
       title=r"$\rho_x$ Delay")
plt.show()

name = f"beta_{beta}-delta_{delta}-gamma_{gamma}-time_{t_fin}"
data_dir = os.path.join(proj_path,"Data","Synthetic","Delay",name)
if not(os.path.exists(data_dir)) : os.mkdir(data_dir)
fig.savefig(os.path.join(data_dir,f"Autocorrx-{name}.png"),format="png",
    facecolor="w",edgecolor="w",bbox_inches="tight")



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


#%%
name = f"beta_{beta}-delta_{delta}-gamma_{gamma}-time_{t_fin}"
data_dir = os.path.join(proj_path,"Data","Synthetic",name)
if not(os.path.exists(data_dir)) : os.mkdir(data_dir)
fig.savefig(os.path.join(data_dir,f"Synthetic-{name}.png"),format="png",
    facecolor="w",edgecolor="w",bbox_inches="tight")

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
df["Cl"] = lib.Cl(df["x"].values,df["y"].values,df["theta"].values,lib.phtrail,param,ks)
df["Cr"] = lib.Cr(df["x"].values,df["y"].values,df["theta"].values,lib.phtrail,param,ks)
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