#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import set_num_threads
import pandas as pd
import sys
sys.path.append('../')
import lib_model as lib

def kl(p,q,binsp,binsq):
    assert(np.all(binsp==binsq))
    p = np.where(p>0)[0]
    q = np.where(q>0)[0]
    idxs = np.intersect1d(p,q)
    p = n1[idxs]
    q = n2[idxs]
    return np.sum(p*np.log(p/q))


#%%

numthreads = 12
set_num_threads(numthreads)
print("Using", numthreads)

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(proj_path,'Estils','plots.mplstyle')) #styles file
proj_path = os.path.split(proj_path)[0] 
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# %%
data_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_nothetarnage_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))

#%%
#Generate a trajectory from the data #TODO: substitute this step for actual data one working.

t_fin = 200
dwr = 1 
v = 6
l = 12.8
phi = 0.95
Mu = 0.0
Sigma = 50
th0 = 1.0
delta = 0.1
beta = 1.5*0.5
gamma = 10

#Model extended
#param = np.array([v,Mu,th0,Sigma,l,phi,delta]) #"known" model parameters
#ks = np.array([beta,gamma]) #This is what we want to inffer!

#Regular
param = np.array([v,Mu,th0])
ks = np.array([beta, delta, Sigma,l,phi])



# %%

#Simulation setup.
h = 0.01 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
Ntraj = 5000 
filtered_ids = datadf.groupby("id_traj")["Time"].max()
filtered_ids = filtered_ids[filtered_ids > 100].index
nindata = len(filtered_ids)

#result_df = datadf[(datadf["id_traj"].isin(filtered_ids)) & (datadf["Time"] == 0)][["id_traj", "x", "y","theta"]]
result_df = datadf[(datadf["id_traj"].isin(filtered_ids))][["id_traj", "Time", "x", "y", "theta", "$|v|$"]]

#%%


names = ["Time","x","y","theta","dif","vx","vy","v","id_traj"]
df = pd.DataFrame(columns=names)
counter = 0
for i in range(Ntraj):
    if counter >= nindata: break
    #CI extended
    dfinit = result_df[result_df["id_traj"] == filtered_ids[counter]]
    x0 = dfinit["x"].iloc[0]
    y0 = dfinit["y"].iloc[0]
    th_ic0 = dfinit["theta"].iloc[0]
    t_fin = dfinit["Time"].max()
    v = dfinit["$|v|$"].mean()
    param[0] = v
    #th_ic0 = np.random.uniform(0,np.pi/2)
    #ci = np.array([0,0,th_ic0,0,0]) 
    #ci[3] = lib.Cl(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
    #ci[4] = lib.Cr(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
    #CI regular
    ci = np.array([x0,y0,th_ic0])
    data = lib.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,1)
    xindx = np.arange(0,len(ci),len(ci))
    yindx = np.arange(1,len(ci),len(ci))
    thindx= np.arange(2,len(ci),len(ci))
    j=0
    if np.max(data[xindx[j]]) < 50 and np.min(data[xindx[j]]) > -50:
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
        counter +=1
# %%
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(7,6))
n1,bins1,_ = ax.hist(datadf["x"],density=True,bins=50)
n2,bins2,_ = ax.hist(df["x"],density=True,alpha=0.75,bins=bins1)
# %%
# %%

# %%
betas = [0.5,0.75,1.5,2,5]
sigmas = [13,35,50,75,150]
sigma = 50
fig,ax = plt.subplots(ncols=len(betas),nrows=len(sigmas),figsize=(7*len(betas),6*len(sigmas)))
for m,beta in enumerate(betas):
    print(beta)
    for n,sigma in enumerate(sigmas):
        print(f"beta = {beta}, sigma = {sigma}")
        param = np.array([v,Mu,th0])
        ks = np.array([beta, delta, sigma,l,phi])
        
        #param = np.array([v,Mu,th0,sigma,l,phi,delta]) #"known" model parameters
        #ks = np.array([beta,gamma]) #This is what we want to inffer!

        names = ["Time","x","y","theta","dif","vx","vy","v","id_traj"]
        df = pd.DataFrame(columns=names)
        counter = 0
        for i in range(Ntraj):
            if counter >= nindata: break
            #CI extended
            dfinit = result_df[result_df["id_traj"] == filtered_ids[counter]]
            x0 = dfinit["x"].iloc[0]
            y0 = dfinit["y"].iloc[0]
            th_ic0 = dfinit["theta"].iloc[0]
            t_fin = dfinit["Time"].max()
            v = dfinit["$|v|$"].mean()
            param[0] = v
            #th_ic0 = np.random.uniform(0,np.pi/2)
            #ci = np.array([x0,y0,th_ic0,0,0]) 
            #ci[3] = lib.Cl(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
            #ci[4] = lib.Cr(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
            #CI regular
            ci = np.array([x0,y0,th_ic0])
            data = lib.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,1)
            xindx = np.arange(0,len(ci),len(ci))
            yindx = np.arange(1,len(ci),len(ci))
            thindx= np.arange(2,len(ci),len(ci))
            j=0
            if np.max(data[xindx[j]]) < 50 and np.min(data[xindx[j]]) > -50:
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
                counter +=1

        n1,bins1,_ = ax[m][n].hist(datadf["x"],density=True,bins=50)
        n2,bins2,_ = ax[m][n].hist(df["x"],density=True,alpha=0.75,bins=bins1)

        ax[m][n].text(0.6,0.8,"kl = {:.3f}".format(kl(n1,n2,bins1,bins2)),transform=ax[m][n].transAxes)
        ax[m][n].text(0.2,0.8,f"b = {beta}",transform=ax[m][n].transAxes)
        ax[m][n].text(0.2,0.6,f"s = {sigma}",transform=ax[m][n].transAxes)
# %%
fig.savefig(os.path.join(proj_path,"Data","Comparisons","Beta_sigma_v.jpg"))
# %%
