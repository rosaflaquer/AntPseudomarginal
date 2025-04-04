#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import set_num_threads
import pandas as pd
import sys
sys.path.append('../')
import lib_model_extended as lib
import json

def kl(p,q,binsp,binsq):
    assert(np.all(binsp==binsq))
    p_2 = np.where(p>0)[0]
    q_2 = np.where(q>0)[0]
    idxs = np.intersect1d(p_2,q_2)
    p = p[idxs]
    q = q[idxs]
    return np.sum(p*np.log(p/q))


#%%

numthreads = 14
set_num_threads(numthreads)
print("Using", numthreads)

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
#plt.style.use(os.path.join(proj_path,'Estils','plots.mplstyle')) #styles file
#proj_path = os.path.split(proj_path)[0] 
#prop_cycle = plt.rcParams['axes.prop_cycle']
#colors = prop_cycle.by_key()['color']

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
Sigma = 75
th0 = 1.0
delta = 0.175
beta = 1.5*0.5
gamma = 10

#Model extended
param = np.array([v,Mu,th0,Sigma,l,phi,delta]) #"known" model parameters
ks = np.array([beta,gamma]) #This is what we want to inffer!

#Simulation setup.
h = 0.1 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
filtered_ids = datadf.groupby("id_traj")["Time"].max()
filtered_ids = filtered_ids[filtered_ids > 100].index
nindata = len(filtered_ids)
max_trials_traj = 1000 
Ntraj = nindata*max_trials_traj

#result_df = datadf[(datadf["id_traj"].isin(filtered_ids)) & (datadf["Time"] == 0)][["id_traj", "x", "y","theta"]]
result_df = datadf[(datadf["id_traj"].isin(filtered_ids))][["id_traj", "Time", "x", "y", "theta", "$|v|$"]]

#%%
betas = [0.5,0.75,1.5,2,5,7.5,15]
gammas = [0.01,0.025,0.05,0.1,0.25,0.5,1,5,10] #
sigma = Sigma
for delta in [0.175]:
    print(delta)
    out_folders = ["Data","Comparisons","Convergence","Delay",f"delta_{delta}"]
    out_path = proj_path
    for folder in out_folders:
        out_path = os.path.join(out_path,folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    fig,ax   = plt.subplots(nrows=len(betas),ncols=len(gammas),figsize=(7*len(betas),6*len(gammas)))
    fig2,ax2 = plt.subplots(nrows=len(betas),ncols=len(gammas),figsize=(7*len(betas),6*len(gammas)))
    for m,beta in enumerate(betas):
        print(beta)
        for n,gamma in enumerate(gammas):
            print(f"beta = {beta}, gamma = {gamma} ########################################")
            
            param = np.array([v,Mu,th0,sigma,l,phi,delta]) #"known" model parameters
            ks = np.array([beta,gamma]) #This is what we want to inffer!

            names = ["Time","x","y","theta","dif","vx","vy","v","id_traj"]
            df = pd.DataFrame(columns=names)
            counter = 0
            deleted = 0
            ids_left = filtered_ids.unique()
            nindata = len(ids_left)
            dict_tries = {}
            deleted_ids = None
            converged_ids = None
            for iid in ids_left:
                dict_tries[iid] = 0
            for i in range(Ntraj):
                if i%1000==0: print(f"beta = {beta}, gamma = {gamma}, iteration = {i}, counter = {counter}")
                if counter + deleted >= nindata: break
                if len(ids_left) > 1:
                    id_idx = np.random.randint(0,len(ids_left)-1)
                else: id_idx = 0
                iid =  ids_left[id_idx]
                dfinit = result_df[result_df["id_traj"] == iid]
                dict_tries[iid] += 1
                if dict_tries[iid] >= max_trials_traj:
                    ids_left = np.delete(ids_left,id_idx)
                    deleted += 1
                    if deleted_ids is None:
                        deleted_ids = str(iid)
                    else:
                        deleted_ids = deleted_ids + "," + str(iid)
                    continue
                x0 = dfinit["x"].iloc[0]
                y0 = dfinit["y"].iloc[0]
                th_ic0 = dfinit["theta"].iloc[0]
                t_fin = dfinit["Time"].max()
                v = dfinit["$|v|$"].mean()
                param[0] = v
                ci = np.array([0,0,th_ic0,0,0]) 
                ci[3] = lib.Cl(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
                ci[4] = lib.Cr(ci[0],ci[1],ci[2],lib.phtrail,param,ks)
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
                    df_temp["id_traj"] = ids_left[id_idx]
                    df = pd.concat([df,df_temp],ignore_index=True)
                    del(df_temp)
                    counter +=1
                    ids_left = np.delete(ids_left,id_idx)
                    if converged_ids is None:
                        converged_ids = str(iid)
                    else:
                        converged_ids = converged_ids + "," + str(iid)
            dict_tries["Total tries"] = i
            dict_tries["Total deleted"] = deleted
            dict_tries["Total converged"] = counter
            dict_tries["Which deleted"] = str(deleted_ids)
            dict_tries["Which converged"] = str(converged_ids)
            not_converged = None
            for iid in ids_left:
                if not_converged is None:
                    not_converged = str(iid) 
                else:
                    not_converged = not_converged + "," + str(iid)
            dict_tries["Which left"] = not_converged
            try:
                n1_x,bins1_x,_ = ax[m][n].hist(datadf["x"],density=True,bins=50)
                n2_x,bins2_x,_ = ax[m][n].hist(df["x"],density=True,alpha=0.75,bins=bins1_x)

                ax[m][n].text(0.6,0.8,"kl = {:.3f}".format(kl(n1_x,n2_x,bins1_x,bins2_x)),
                            transform=ax[m][n].transAxes)
                ax[m][n].text(0.2,0.8,f"b = {beta}",transform=ax[m][n].transAxes)
                ax[m][n].text(0.2,0.6,f"g = {gamma}",transform=ax[m][n].transAxes)
                ax[m][n].text(0.2,0.4,"c = {:.3f}".format(counter/i),transform=ax[m][n].transAxes)

                n1_y,bins1_y,_ = ax2[m][n].hist(datadf["y"],density=True,bins=50)
                n2_y,bins2_y,_ = ax2[m][n].hist(df["y"],density=True,alpha=0.75,bins=bins1_y)
                ax2[m][n].text(0.6,0.8,"kl = {:.3f}".format(kl(n1_y,n2_y,bins1_y,bins2_y)),
                            transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.2,0.8,f"b = {beta}",transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.2,0.6,f"g = {gamma}",transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.2,0.4,"c = {:.3f}".format(counter/i),transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.6,0.4,f"d = {deleted}",transform=ax2[m][n].transAxes)
                
                name = f"Beta_{beta}_gamma_{gamma}_sigma_{sigma}_delta{delta}"

                np.savez(os.path.join(out_path, name + "_v_x.npz"),
                         n1_x=n1_x, bins1_x=bins1_x, n2_x=n2_x, bins2_x=bins2_x,)

                np.savez(os.path.join(out_path, name + "_v_y.npz"),
                         n1_x=n1_y, bins1_x=bins1_y, n2_x=n2_y, bins2_x=bins2_y,)
                
                with open(os.path.join(out_path, name + "_dict_tries.json"), "w") as f:
                    json.dump(dict_tries, f)
            except Exception as e:
                print(f"Error in building the histogram: {e}")
                pass

    fig.savefig(os.path.join( out_path,f"Beta_gamma-sigma_{sigma}_delta_{delta}_v_x.jpg"))
    fig2.savefig(os.path.join(out_path,f"Beta_gamma-sigma_{sigma}_delta_{delta}_v_y.jpg"))


# %%
