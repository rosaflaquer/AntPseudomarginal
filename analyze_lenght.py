#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import scipy.stats as sps

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
#main_dir = os.path.join(proj_path,"Data","Fits","Cut")
folders = os.listdir(main_dir)
converged = []
id_traj = datadf["id_traj"].unique()
dict_params_total = {
        "beta":   {"mean":[],"std":[],"hatR":[],"folder":[],"id_traj":[]},
        "delta":  {"mean":[],"std":[],"hatR":[],"folder":[],"id_traj":[]},    
        "sigma":  {"mean":[],"std":[],"hatR":[],"folder":[],"id_traj":[]},    
        "l":      {"mean":[],"std":[],"hatR":[],"folder":[],"id_traj":[]},
        "phi":    {"mean":[],"std":[],"hatR":[],"folder":[],"id_traj":[]},
    }
dict_params = {
    "beta":     0,
    "delta":    1,    
    "sigma":    2,    
    "l":        3,
    "phi":      4,
    }

#%%

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
            dict_params_total[parameter]["folder"].append(name)
            dict_params_total[parameter]["id_traj"].append(id_tr)
# %%

for parameter in dict_params.keys():
    dict_params_total[parameter]["mean"] = np.array(dict_params_total[parameter]["mean"])
    dict_params_total[parameter]["std"] = np.array(dict_params_total[parameter]["std"])
    dict_params_total[parameter]["hatR"] = np.array(dict_params_total[parameter]["hatR"])
    dict_params_total[parameter]["folder"] = np.array(dict_params_total[parameter]["folder"])
    dict_params_total[parameter]["id_traj"] = np.array(dict_params_total[parameter]["id_traj"])

# %%
threshold = 1.1
parameter = "l"
xx = dict_params_total[parameter]["mean"]
mask_x = np.where(dict_params_total[parameter]["hatR"]<threshold)
xx = xx[mask_x]
iids = dict_params_total[parameter]["id_traj"][mask_x]

lengths = []
vels = []
for iid in iids:
    df_traj = datadf[datadf["id_traj"]==iid]
    lengths.append(df_traj["diff"].sum())
    vels.append(df_traj["$|v|$"].mean())

lengths = np.array(lengths)
vels = np.array(vels)

fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(18,10))
axs[0].hist2d(lengths,xx,cmap="binary")
axs[0].scatter(lengths,xx,color=colors[2])
axs[0].set(xlabel="len",ylabel=parameter)
pc,pv = sps.pearsonr(xx,lengths)
axs[0].text(0.8,0.8,"{:.2f}".format(pc),transform=axs[0].transAxes,size=50) 
axs[1].hist2d(vels,xx,cmap="binary")
axs[1].scatter(vels,xx,color=colors[2])
axs[1].set(xlabel=r"$v$",ylabel=parameter)
pc,pv = sps.pearsonr(xx,vels)
axs[1].text(0.8,0.8,"{:.2f}".format(pc),transform=axs[1].transAxes,size=50)


path_fig = os.path.split(main_dir)[0]
fig.savefig(os.path.join(path_fig,f"{parameter}_lenvel_{threshold}.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
# %%
