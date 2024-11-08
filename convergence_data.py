#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lib_MCMC import *
import lib_model

dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# %%
#1) Load observations
name = "Traj_04_oct_203_9.0"
data_dir = os.path.join(proj_path,"Data","Fits",name)
traj_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(traj_dir,data_file))
id_traj = datadf["id_traj"].unique()
Ntrajs = len(id_traj)
id = name[5:]
day_traj = datadf[datadf["id_traj"]==id]
print(len(day_traj))
data = np.zeros((4,len(day_traj)))
data[0] = day_traj["x"].values     
data[1] = day_traj["y"].values     
data[2] = day_traj["theta"].values 
data[3] = day_traj["Time"].values
#%%

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.plot(data[0],data[1])
ax.set(xlabel=r"$x$",ylabel=r"$y$")
filename = f"traj_{name}.png"
#fig.savefig(os.path.join(data_dir,filename),format="png",
#            facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
file_name = "Chains-"

df = pd.read_csv(os.path.join(data_dir,file_name+name+".dat"))
nparam = 2
#C = int(len(df.columns)/2)
C = int(len(df.columns)/4)
M = len(df)
chains = []
for i in range(C):
    chains.append([df[f"beta_{i}"].values,df[f"delta_{i}"].values])
print(len(chains[0]))
hatR,Se = convergence(chains,nparam,C,M)
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
filename = file_name + f"beta-{name}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
filename = file_name + f"delta-{name}.png"
fig1.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
#%%

def plot(xx,xlabel,nbins,filename):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
    height, _, _ = ax.hist(xx,bins=nbins,density=True)
    m = np.mean(xx)
    sd = np.std(xx)
    ci = [m-sd,m+sd]
    ylims = [0, height.max()*(1+0.1)]
    ax.vlines(m,ylims[0],ylims[1],color="black")
    ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35) 
    ax.set(xlabel=xlabel,ylim=ylims)
    plt.show()
    fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

nbins = 30
plot(betas,r"$\beta$",nbins,f"Distr_beta-traj_{name}.png")
plot(deltas,r"$\delta$",nbins,f"Distr_delta-traj_{name}.png")

#%%


bins = 50
fig = plt.figure(figsize=(15,12))
ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
ax.set(xlabel=r"$\beta$",ylabel=r"$\delta$")
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
#ax.scatter(betas, deltas,s=5)
ax.hist2d(betas,deltas,bins=25,cmap="binary")

ax.axvline(np.mean(betas),ls="--")
mb = np.mean(betas)
sdb = np.std(betas)
ax.fill_betweenx([deltas.min(),deltas.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 

height, _,_ = ax_histx.hist(betas, bins=bins, density=True,color="black")
ax_histx.axvline(np.mean(betas),ls="--")
ax_histx.fill_betweenx([0,height.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 

ax.axhline(np.mean(deltas),ls="--")
md = np.mean(deltas)
sdd = np.std(deltas)
ax.fill_between([betas.min(),betas.max()], md - sdd, md + sdd, color=colors[0], alpha=0.15) 

height, _,_ = ax_histy.hist(deltas, bins=bins, density=True, orientation='horizontal',color="black")
ax_histy.axhline(np.mean(deltas),ls="--")
ax_histy.fill_between([0,height.max()],  md - sdd, md + sdd, color=colors[0], alpha=0.15) 

filename = f"Distr-traj_{name}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
        facecolor="w",edgecolor="w",bbox_inches="tight")


#%%

#Posterior checks

def sqdispl(datax,datay):
    if len(np.shape(datax)) > 1:
        return np.sum(np.sqrt((datax[1:]-datax[:-1])**2+(datay[1:]-datay[:-1])**2),axis=1)
    else:
        return np.sum(np.sqrt((datax[1:]-datax[:-1])**2+(datay[1:]-datay[:-1])**2))

def count_zero_crossings(data):
    signs = np.sign(data)
    diff_signs = np.diff(signs)
    if len(np.shape(data)) > 1:
        return np.count_nonzero(diff_signs,axis=1)
    else: 
        return np.count_nonzero(diff_signs)


h = 0.01 
v,l,phi,Mu,Sigma,th0 = day_traj["$|v|$"].mean(),13,0.9,0.0,6.5,1.0
param = np.array([v,l,phi,Mu,Sigma,th0 ])
icon = np.array([data[0,0],data[1,0],data[2,0]])
t_fin = data[3,-1]
Dt = data[3,1]-data[3,0]
print(t_fin,Dt)
iwr = int(Dt/h)
Nt = int(t_fin/h) + 1*iwr
Ntraj_ic = 5000
ndraws = 3
xidxs = [i for i in range(0,Ntraj_ic,3)]
yidxs = [i for i in range(1,Ntraj_ic,3)]
#%%
fig,axs=plt.subplots(ncols=3,nrows=3,figsize=(11*3.5,6*(ndraws+0.5)))
for i in range(ndraws):
    b_distr = rv_from_data(betas)
    d_distr = rv_from_data(deltas)
    print(b_distr,d_distr)
    ks = np.array([b_distr, d_distr])
    data_pred = lib_model.multiple_traj(icon,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj_ic)
    
    ax = axs[i][0]
    for j in range(0,len(data_pred),3):
        ax.plot(data_pred[j],data_pred[(j+1)],alpha=0.25)

    ss = np.std(data_pred[xidxs],axis=0)
    xm = np.mean(data_pred[xidxs],axis=0)
    ym = np.mean(data_pred[yidxs],axis=0)
    ax.plot(xm,ym,color="white")
    ax.fill_betweenx(ym,xm-ss,xm+ss,color="black",alpha=0.9)
    ax.plot(data[0,:],data[1,:],color="black")
    ax.set(xlabel=r"$x$",ylabel=r"$y$")
    ax.text(0.75,0.8,
            r"$\beta^*$={:.2f}".format(b_distr) +"\n" + r"$\delta^*$={:.2f}".format(d_distr),
            transform=ax.transAxes)

    sqd_pred = sqdispl(data_pred[xidxs],data_pred[yidxs])
    sqd = sqdispl(data[0],data[1])

    cros_pred = count_zero_crossings(data_pred[xidxs])
    cros =  count_zero_crossings(data[0])

    ax = axs[i][1]
    height, _,_ = ax.hist(sqd_pred,bins=50,density=True)
    m = np.mean(sqd_pred)
    sd = np.std(sqd_pred)
    ci = [m-sd,m+sd]
    ylims = [0, height.max()*(1+0.1)]
    ax.vlines(m,ylims[0],ylims[1],color="black")
    ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35) 
    ax.vlines(sqd,ylims[0],ylims[1],color="black",ls="--")
    ax.set(yscale="log",xlabel="SD")
    ax.text(0.75,0.8,
            r"$\beta^*$={:.2f}".format(b_distr) +"\n" + r"$\delta^*$={:.2f}".format(d_distr),
            transform=ax.transAxes)

    ax = axs[i][2]
    height, _,_ = ax.hist(cros_pred,bins=50,density=True)
    m = np.mean(cros_pred)
    sd = np.std(cros_pred)
    ci = [m-sd,m+sd]
    ylims = [0, height.max()*(1+0.1)]
    ax.vlines(m,ylims[0],ylims[1],color="black")
    ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35)        
    ax.vlines(cros,ylims[0],ylims[1],color="black",ls="--")
    ax.set(yscale="log",xlabel="Crossings")
    ax.text(0.75,0.8,
            r"$\beta^*$={:.2f}".format(b_distr) +"\n" + r"$\delta^*$={:.2f}".format(d_distr),
            transform=ax.transAxes)

plt.show()

filename = f"Posterior_check-traj_{name}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
#%%
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
for j in range(0,len(data_pred),3):
    if j > 30*3: break
    ax.plot(data_pred[j],data_pred[(j+1)],alpha=0.25)
ax.plot(data[0,:],data[1,:],color="black")