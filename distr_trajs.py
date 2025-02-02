#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#%%
traj_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_nomin_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(traj_dir,data_file))
main_dir = os.path.join(proj_path,"Data","Fits","Fitsnomin","mcmc","Data","Fits")
folders = os.listdir(main_dir)
converged = []
id_traj = datadf["id_traj"].unique()
cf = open(os.path.join(main_dir,f"pars.dat"),"a")
dict_params_total = {
        "beta":   {"mean":[],"std":[],"hatR":[]},
        "delta":  {"mean":[],"std":[],"hatR":[]},    
        "sigma":  {"mean":[],"std":[],"hatR":[]},    
        "l":      {"mean":[],"std":[],"hatR":[]},
        "phi":    {"mean":[],"std":[],"hatR":[]},
    }
dict_params = {
    "beta":     0,
    "delta":    1,    
    "sigma":    2,    
    "l":        3,
    "phi":      4,
    }
for name in folders:
    data_dir = os.path.join(main_dir,name)
    for tr in id_traj:
       
        is_segment = tr.find(".0_s")
        if is_segment > 0: id_folder = tr[:is_segment+1]
        else: id_folder = tr
        if id_folder != name[5:] : 
            continue
        else : id_tr = tr
        print(name,id_folder,id_tr)
        
        file_name = "Chains-Traj_"
        df = pd.read_csv(os.path.join(data_dir,file_name+id_tr+".dat"))
        nparam = len(dict_params)
        C = int(len(df.columns)/(nparam+2))
        M = len(df)
        
        file_name = f"log_chains-Traj_{id_tr}.dat"
        with open(os.path.join(data_dir,file_name),"r") as f:
            for i in range(2*C):
                f.readline()
            f.readline()
            array_str = f.readline().split('=')[1].strip()
            hatR = np.fromstring(array_str.strip('[]'), sep=' ')
            array_str = f.readline().split('=')[1].strip()
            ESS = np.fromstring(array_str.strip('[]'), sep=' ')
        
        for parameter in dict_params.keys():
            idx_par = dict_params[parameter]
            columns = [f"par{idx_par}_{i}" for i in range(C)] #columns for each parameter
            stacked = mean_value = df[columns].stack()
            dict_params_total[parameter]["mean"].append(stacked.mean())
            dict_params_total[parameter]["std"].append(stacked.std())
            dict_params_total[parameter]["hatR"].append(hatR[idx_par])
# %%
threshold = 1.01
for parameter in dict_params.keys():
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(11,6))
    fig.suptitle(r"${}$".format(parameter))
    for i,title in enumerate(["mean","std"]):
        xx = np.array(dict_params_total[parameter][title])
        mask = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
        axs[i].hist(xx[mask],density=True)
        axs[i].set_title(title)
    plt.show()
    fig.savefig(os.path.join(os.path.split(main_dir)[0],parameter+f"_thresh_{threshold}.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

# %%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
parameter = "beta"
xx = np.array(dict_params_total[parameter]["mean"])
mask_x = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
parameter = "phi"
yy = np.array(dict_params_total[parameter]["mean"])
mask_y = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
mask = np.intersect1d(mask_x,mask_y)
ax.hist2d(xx[mask],yy[mask],cmap="binary",density=True)
ax.set(xlabel=r"$\beta$",ylabel=r"$\phi$")
fig.savefig(os.path.join(os.path.split(main_dir)[0],f"beta_phi_thresh_{threshold}.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

# %%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
parameter = "delta"
xx = np.array(dict_params_total[parameter]["mean"])
mask_x = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
parameter = "sigma"
yy = np.array(dict_params_total[parameter]["mean"])
mask_y = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
mask = np.intersect1d(mask_x,mask_y)
ax.hist2d(xx[mask],yy[mask],cmap="binary",density=True)
ax.set(xlabel=r"$\delta$",ylabel=r"$\sigma$")
fig.savefig(os.path.join(os.path.split(main_dir)[0],f"delta_sigma_thresh_{threshold}.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

# %%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
parameter = "sigma"
xx = np.array(dict_params_total[parameter]["mean"])
mask_x = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
parameter = "phi"
yy = np.array(dict_params_total[parameter]["mean"])
mask_y = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
mask = np.intersect1d(mask_x,mask_y)
ax.hist2d(xx[mask],yy[mask],cmap="binary",density=True)
ax.set(xlabel=r"$\sigma$",ylabel=r"$\phi$")
fig.savefig(os.path.join(os.path.split(main_dir)[0],f"sigma_phi_thresh_{threshold}.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

# %%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
parameter = "l"
xx = np.array(dict_params_total[parameter]["mean"])
mask_x = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
parameter = "phi"
yy = np.array(dict_params_total[parameter]["mean"])
mask_y = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
mask = np.intersect1d(mask_x,mask_y)
ax.hist2d(xx[mask],yy[mask],cmap="binary",density=True)
ax.set(xlabel=r"$l$",ylabel=r"$\phi$")
fig.savefig(os.path.join(os.path.split(main_dir)[0],f"l_phi_thresh_{threshold}.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

# %%
