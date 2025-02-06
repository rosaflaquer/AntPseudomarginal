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
threshold = 1.01
for parameter in dict_params.keys():
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(11,6))
    fig.suptitle(r"${}$".format(parameter))
    for i,title in enumerate(["mean","std"]):
        xx = np.array(dict_params_total[parameter][title])
        mask = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
        axs[i].hist(xx[mask],density=True,bins=15)
        axs[i].set_title(title)
    plt.show()
    #fig.savefig(os.path.join(os.path.split(main_dir)[0],parameter+f"_thresh_{threshold}.png"),format="png",
    #        facecolor="w",edgecolor="w",bbox_inches="tight")

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
#fig.savefig(os.path.join(os.path.split(main_dir)[0],f"beta_phi_thresh_{threshold}.png"),format="png",
#            facecolor="w",edgecolor="w",bbox_inches="tight")

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
# Plot non converging trajectories

threshold = 1.01
threshold_pause = 1
for parameter in dict_params.keys():
    fig,ax = plt.subplots(ncols=3,nrows=1,figsize=(16,7))
    ax[0].set(xlabel=r"$x$",ylabel=r"$y$")
    ax[1].set(xlabel=r"$|v|$")
    ax[2].set(xlabel="Pause length")
    fig.suptitle(r"${}$".format(parameter))
    mask = np.where(np.array(dict_params_total[parameter]["hatR"])>threshold)
    ids = np.array(dict_params_total[parameter]["id_traj"])[mask]
    for id_traj in ids:
        day_traj = datadf[datadf["id_traj"]==id_traj]
        ax[0].plot(day_traj["x"],day_traj["y"])
    mask_ids = datadf['id_traj'].isin(ids)
    ax[1].hist(datadf[mask_ids]["$|v|$"],density=True)
    subset = datadf[mask_ids]
    pauses = subset[subset["$|v|$"] < 0.9]
    times = pauses.groupby("id_traj")["diff"].sum().values
    times = times/pauses.groupby("id_traj")["Time"].max().values
    ax[2].hist(times,density=True)
    ax[2].text(0.75, 0.8, '{:.2f}'.format(len(times)/len(ids)), transform=ax[2].transAxes)
    plt.show()
    fig.savefig(os.path.join(os.path.split(main_dir)[0],parameter+f"_thresh_{threshold}_No_conv.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

# %%

# Plot non converging trajectories
out_path = os.path.split(main_dir)[0]
out_path = os.path.join(out_path,"Trajs_NoConv")
if not(os.path.exists(out_path)): os.mkdir(out_path)
for parameter in dict_params.keys():
    mask = np.where(np.array(dict_params_total[parameter]["hatR"])>threshold)
    ids = np.array(dict_params_total[parameter]["id_traj"])[mask]
    for id_traj in ids:
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(16,7))
        ax.set(xlabel=r"$x$",ylabel=r"$y$",title=f"{parameter}, {id_traj}")
        day_traj = datadf[datadf["id_traj"]==id_traj]
        cm = ax.scatter(day_traj["x"],day_traj["y"],c=day_traj["$|v|$"],vmax=5,vmin=0)
        ax.axvline(0,color="black")
        plt.colorbar(cm,ax=ax)
        plt.show()
        fig_path = os.path.join(out_path,parameter)
        if not(os.path.exists(fig_path)): os.mkdir(fig_path)
        fig.savefig(os.path.join(fig_path,parameter+f"{id_traj}_thresh_{threshold}_No_conv.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
# %%
parameter = "beta"
trajs_ident = []
for i in range(len(dict_params_total[parameter]["id_traj"])):
    b = dict_params_total[parameter]["mean"][i]
    rh =  dict_params_total[parameter]["hatR"][i]
    if b > 0.5 or rh < threshold: continue
    id_traj = dict_params_total[parameter]["id_traj"][i]
    trajs_ident.append(id_traj)
    print(id_traj,b)
    fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(16,7))
    fig.suptitle(id_traj)
    day_traj = datadf[datadf["id_traj"]==id_traj]
    cm = ax[0].scatter(day_traj["x"],day_traj["y"],c=day_traj["$|v|$"],vmax=5,vmin=0)
    ax[0].set(xlabel=r"$x$",ylabel=r"$y$")
    ax[0].axvline(0,color="black")
    ax[1].plot(day_traj["Time"],day_traj["x"])
    ax[1].set(xlabel=r"$t$",ylabel=r"$x$")
    ax[1].axhline(0,color="black")
    plt.show()
# %%
parameter = "beta"
par_nocross = []
par_cros = []
for i in range(len(dict_params_total[parameter]["id_traj"])):
    b = dict_params_total[parameter]["mean"][i]
    rh =  dict_params_total[parameter]["hatR"][i]
    id_traj = dict_params_total[parameter]["id_traj"][i]
    if rh < threshold: continue
    day_traj = datadf[datadf["id_traj"]==id_traj].copy()
    positive_mask = day_traj['x'] < 0
    diff = positive_mask.diff() # Compute the difference of the boolean mask
    changes = (diff == 1).sum() # Count the number of times the series changes from positive to negative
    print(f"Number of changes from positive to negative: {changes}")
    if changes <= 2: par_nocross.append(b)
    else: par_cros.append(b)    
# %%
print(len(par_nocross)/(len(par_nocross) + len(par_cros)))
# %%
fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(11,6))
ax[0].hist(par_cros,density=True)
ax[1].hist(par_nocross,density=True)
# %%
len(folders)
# %%
converged = {}
for parameter in dict_params.keys():
    mask = np.where(np.array(dict_params_total[parameter]["hatR"])<threshold)
    converged[parameter] = np.array(dict_params_total[parameter]["id_traj"])[mask]
# %%
arr_0 = converged["beta"]
print(len(arr_0))
for parameter in converged.keys():
    arr_1 = converged[parameter]
    arr_0 = np.intersect1d(arr_0,arr_1)
    print(parameter,len(arr_0),len(arr_1))
# %%
out_path = os.path.split(main_dir)[0]
out_path = os.path.join(out_path,"Trajs_Conv")
if not(os.path.exists(out_path)): os.mkdir(out_path)
for id_traj in arr_0:
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(16,7))
    ax.set(xlabel=r"$x$",ylabel=r"$y$",title=f"{id_traj}")
    day_traj = datadf[datadf["id_traj"]==id_traj]
    cm = ax.scatter(day_traj["x"],day_traj["y"],c=day_traj["$|v|$"],vmax=5,vmin=0)
    ax.axvline(0,color="black")
    plt.colorbar(cm,ax=ax)
    fig.savefig(os.path.join(out_path,f"{id_traj}_thresh_{threshold}_Conv.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
    plt.show()
# %%
out_path = os.path.split(main_dir)[0]
out_path = os.path.join(out_path,"Trajs_NoConv")
if not(os.path.exists(out_path)): os.mkdir(out_path)
for id_traj in dict_params_total["beta"]["id_traj"]:
    if id_traj in arr_0: continue
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(16,7))
    ax.set(xlabel=r"$x$",ylabel=r"$y$",title=f"{id_traj}")
    day_traj = datadf[datadf["id_traj"]==id_traj]
    cm = ax.scatter(day_traj["x"],day_traj["y"],c=day_traj["$|v|$"],vmax=5,vmin=0)
    ax.axvline(0,color="black")
    plt.colorbar(cm,ax=ax)
    fig.savefig(os.path.join(out_path,f"{id_traj}_thresh_{threshold}_Conv.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
    plt.show()
#%%
