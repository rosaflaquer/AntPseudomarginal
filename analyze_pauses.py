#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#%%

traj_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(traj_dir,data_file))
main_dir = os.path.join(proj_path,"Data","Fits","Fits09","mcmc","Data")
folders_dir = os.path.join(main_dir,"Fits")
folders = os.listdir(folders_dir)

with open(os.path.join(main_dir,"Converged.dat"),"r") as f:
    converged = [line.strip("\n",).strip("Traj_") for line in f.readlines()]


not_converged = []
vstop = 1.2
for folder in folders:
    if not(folder in converged):
        id_traj = folder.strip("Traj_") 
        not_converged.append(id_traj)
        #trajdf = datadf[datadf["id_traj"]==id_traj].copy()
        #if len(trajdf) == 0: continue
        #fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6)) 
        #ax.plot(trajdf["Time"],trajdf["$|v|$"])
        #ax.plot(trajdf["Time"],trajdf["vx"])
        #ax.plot(trajdf["Time"],trajdf["vy"])
        #ax.axhline("0",color="black")
        #ax.set(ylim=[-8,8])
        #plt.show()

datadf["State_pause"] =  np.where(datadf["$|v|$"] < vstop,1,0)
datadf["Change_pause"] = datadf.groupby(["id_traj"])["State_pause"].shift(periods=-1)
datadf["Change_pause"] = abs(datadf["State_pause"]-datadf["Change_pause"])
datadf["Segment_pause"] = datadf.groupby(["id_traj"])["Change_pause"].cumsum()+1

dict_converged = {}
for conv,indx in [("conv",converged),("not_conv",not_converged)]:
    dict_pauses = {}
    iids_trajs = datadf["id_traj"].unique()
    len_movements = datadf.groupby(["id_traj","Segment_pause"])["diff"].sum()
    len_trajs = datadf.groupby(["id_traj"])["diff"].sum()
    total_npause = []
    total_lenpause = []
    total_lenmove = []
    for id_traj in indx:
        val_len_movements = len_movements[id_traj].values
        len_traj = len_trajs[id_traj]
        if datadf[datadf["id_traj"]==id_traj]["State_pause"].iloc[0] == 0: 
            moving, pauses = val_len_movements[::2]/len_traj, val_len_movements[1::2]/len_traj
        else: 
            moving, pauses = val_len_movements[1::2]/len_traj, val_len_movements[::2]/len_traj
        npause = len(pauses) 
        total_npause.append(npause)
        for p in pauses: total_lenpause.append(p)
        for m in moving: total_lenmove.append(m)
        pauses_info = [moving,pauses,npause]
        dict_pauses[id_traj] = pauses_info
    dict_converged[conv] = [dict_pauses,total_lenmove,total_lenpause,total_npause]

#%%
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha=1
nbins=20
for conv in ["conv","not_conv"]:
    ax.hist(dict_converged[conv][1],bins=nbins,label=conv,density=True,alpha=alpha)
    alpha = alpha - 0.25
ax.set(title="Length movement", xlabel=r"$t_{move}/t_{total}$")
ax.legend()
plt.show()
fig.savefig(os.path.join(main_dir,"Length_movement.png"),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha=1
nbins=20
for conv in ["conv","not_conv"]:
    ax.hist(dict_converged[conv][2],bins=nbins,label=conv,density=True,alpha=alpha)
    alpha = alpha - 0.25
ax.set(title="length pauses", xlabel=r"$t_{pause}/t_{total}$")
ax.legend()
plt.show()
fig.savefig(os.path.join(main_dir,"Length_pause.png"),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha=1
nbins=10
for conv in ["conv","not_conv"]:
    ax.hist(dict_converged[conv][3],bins=nbins,label=conv,density=True,alpha=alpha)
    alpha = alpha - 0.25
ax.legend()
ax.set(title="Number pauses", xlabel="Npauses")
plt.show()
fig.savefig(os.path.join(main_dir,"Number_pauses.png"),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")
#%%
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha = 1
nbins = 10
for conv,indx in [("conv",converged),("not_conv",not_converged)]:
    vels = []
    vels_segments = []
    for id_traj in indx:
        trajdf = datadf[datadf["id_traj"]==id_traj]
        vmin = trajdf["$|v|$"].min()
        vv = (trajdf["$|v|$"].values - vmin)/vmin
        for v in vv: vels.append(v)
        trajdf = trajdf[trajdf["State_pause"] < 1]
        segments = trajdf["Segment_pause"].unique()
        for segment in segments:
            segdf = trajdf[trajdf["Segment_pause"]==segment]
            vmin = segdf["$|v|$"].min()
            vv = (segdf["$|v|$"].values - vmin)/vmin
            for v in vv: vels_segments.append(v)
    ax.hist(vels,bins=nbins,label=conv,density=True,alpha=alpha,edgecolor="black")
    alpha = alpha - 0.2
    ax.hist(vels_segments,bins=nbins,label=conv+"_seg",density=True,alpha=alpha,edgecolor="black", hatch='/')
    alpha = alpha - 0.2
ax.set(xlabel=r"$|v|/|v|_{min} - 1$",xlim=[0,6])
ax.legend()
fig.savefig(os.path.join(main_dir,"v_vmin.png"),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha = 1
nbins = 10
for conv,indx in [("conv",converged),("not_conv",not_converged)]:
    vels = []
    vels_segments = []
    for id_traj in indx:
        trajdf = datadf[datadf["id_traj"]==id_traj]
        vmin = trajdf["$|v|$"].min()
        vmax = trajdf["$|v|$"].max()
        vv = (trajdf["$|v|$"].values - vmin)/vmax
        for v in vv: vels.append(v)
        trajdf = trajdf[trajdf["State_pause"] < 1]
        segments = trajdf["Segment_pause"].unique()
        for segment in segments:
            segdf = trajdf[trajdf["Segment_pause"]==segment]
            vmin = segdf["$|v|$"].min()
            vmax = segdf["$|v|$"].max()
            vv = (segdf["$|v|$"].values - vmin)/vmax
            for v in vv: vels_segments.append(v)
    ax.hist(vels,bins=nbins,label=conv,density=True,alpha=alpha,edgecolor="black")
    alpha = alpha - 0.2
    ax.hist(vels_segments,bins=nbins,label=conv+"_seg",density=True,alpha=alpha,edgecolor="black", hatch='/')
    alpha = alpha - 0.2
ax.set(xlabel=r"$(|v| - |v|_{min})/|v|_{max}$",xlim=[0,1])
ax.legend()
fig.savefig(os.path.join(main_dir,"v_vmin_vmax.png"),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha = 1
nbins = 15
for conv,indx in [("conv",converged),("not_conv",not_converged)]:
    vels = []
    vels_segments = []
    for id_traj in indx:
        trajdf = datadf[datadf["id_traj"]==id_traj]
        vmax = trajdf["$|v|$"].max()
        vv = 1 - trajdf["$|v|$"].values/vmax
        for v in vv: vels.append(v)
        trajdf = trajdf[trajdf["State_pause"] < 1]
        segments = trajdf["Segment_pause"].unique()
        for segment in segments:
            segdf = trajdf[trajdf["Segment_pause"]==segment]
            vmax = segdf["$|v|$"].max()
            vv =  1 - segdf["$|v|$"].values/vmax
            for v in vv: vels_segments.append(v)
    ax.hist(vels,bins=nbins,label=conv,density=True,alpha=alpha,edgecolor="black")
    alpha = alpha - 0.2
    ax.hist(vels_segments,bins=nbins,label=conv+"_seg",density=True,alpha=alpha,edgecolor="black", hatch='/')
    alpha = alpha - 0.2
ax.set(xlabel=r"$1-|v|/|v|_{max}$",xlim=[0,1])
ax.legend()
fig.savefig(os.path.join(main_dir,"v_vmax.png"),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")
#%%
nbins=100
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
alpha = 1
ax.hist(datadf["vx"].abs(),density=True,bins=nbins,alpha=alpha)
alpha = alpha - 0.2
ax.hist(datadf["vy"].abs(),density=True,bins=nbins,alpha=alpha)
alpha = alpha - 0.2
ax.hist(datadf["$|v|$"],   density=True,bins=nbins,alpha=alpha)
ax.set(xticks=np.linspace(0,10,10))
#%%
datadf['id_traj_s'] = datadf['id_traj'] +"_s" + datadf['Segment_pause'].astype(str)
segdf = datadf[datadf["State_pause"]<1].copy()
segdf.dropna(inplace=True)
#%%
print(segdf.columns)
print(len(datadf['id_traj_s'].unique()))
print(len(datadf['id_traj'].unique()))
print(len(segdf['id_traj_s'].unique()))
print(segdf["State_pause"].values.sum())
print(len(datadf),len(segdf),len(segdf)/len(datadf))
#%%
if 'id_traj_s' in segdf.columns :
    segdf['id_traj_original'] = segdf['id_traj']
    segdf['id_traj'] = segdf['id_traj_s']
    segdf.drop(['id_traj_s'],axis=1,inplace=True)
#%%
ids = segdf["id_traj"].unique()
with open(os.path.join(os.getcwd(),"id_trajs_150_nopause.par"),"w") as f:
    for i in ids:
        f.write(i+"\n")
#%%
iid =  segdf['id_traj_original'][0]
print(iid)
is_segment = iid.find("_s")
if is_segment > 0: iid = iid[:is_segment]
print(iid)

#%%
segdf.to_csv(os.path.join(os.path.split(dirname)[0],"Data","Ant_data","Nopause"+data_file),index=False)