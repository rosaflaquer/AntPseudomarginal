#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lib_MCMC import *
import lib_model

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# %%
#1) Load observations
beta, delta, gamma, t_fin = 0.4,0.075,0.7,150
name = f"beta_{beta}-delta_{delta}-gamma_{gamma}-time_{t_fin}"
data_dir = os.path.join(proj_path,"Data","Synthetic",name)
data_file= f"Synthetic-{name}.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))
id_traj = datadf["id_traj"].unique()
Ntrajs = len(id_traj)
len_trajs = len(datadf[datadf["id_traj"]==id_traj[0]])
data = np.zeros((Ntrajs,4,len_trajs))
for i,id in enumerate(id_traj):
    day_traj = datadf[datadf["id_traj"]==id]
    data[i][0] = day_traj["x"].values     
    data[i][1] = day_traj["y"].values     
    data[i][2] = day_traj["theta"].values 
    data[i][3] = day_traj["Time"].values

ntraj_data = len(id_traj)
#%%
idx = 7
df = pd.read_csv(os.path.join(data_dir,f"Chains-traj_{idx}.dat"))
nparam = 2
C = int(len(df.columns)/(nparam+2))
M = len(df)
chains = []
for i in range(C):
    #chains.append([df[f"beta_{i}"].values,df[f"delta_{i}"].values])
    onechain = []
    for j in range(nparam):
        onechain.append(df[f"par{j}_{i}"].values)
    chains.append(onechain)

hatR,Se = convergence(chains,nparam,C,M)
print(hatR,Se)
#%%

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.plot(data[idx,0,:],data[idx,1,:])
ax.set(xlabel=r"$x$",ylabel=r"$y$")
filename = f"traj_{idx}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

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
filename = f"Chains_beta-traj_{idx}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
filename = f"Chains_delta-traj_{idx}.png"
fig1.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
#%%
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
fig1,ax1=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
fig2,ax2=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.set(ylabel=r"$\sigma$",xlabel="MC step")
ax1.set(ylabel=r"$l$",xlabel="MC step")
ax2.set(ylabel=r"$\phi$",xlabel="MC step")
sigmas = np.zeros(M*C)
ls = np.zeros(M*C)
phis = np.zeros(M*C)
for i in range(C):
    ax.plot(chains[i][2])
    ax1.plot(chains[i][3])
    ax2.plot(chains[i][4])
    sigmas[i*M:(i+1)*M] = chains[i][2]
    ls[i*M:(i+1)*M]= chains[i][3]
    phis[i*M:(i+1)*M]= chains[i][4]
plt.show()
filename = f"Chains_sigma-traj_{idx}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
filename = f"Chains_l-traj_{idx}.png"
fig1.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
filename = f"Chains_phi-traj_{idx}.png"
fig2.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
#%%

def plot(xx,xlabel,nbins,filename):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
    height, _, _ = ax.hist(xx,bins=nbins,density=True)
    m = np.mean(xx)
    sd = np.std(xx)
    ci = [m-sd,m+sd]
    ylims = [0, height.max()*(1+0.1)]
    ax.vlines(m,ylims[0],ylims[1],color="black",label=r"$\langle${}$\rangle$ = {:.3f} $\pm$ {:.3f}".format(xlabel,m,sd))
    ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35) 
    ax.set(xlabel=xlabel,ylim=ylims)
    ax.legend()
    #plt.show()
    fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
    return fig, ax

nbins = 30
plot(betas,r"$\beta$",nbins,f"Distr_beta-traj_{idx}.png")
plot(deltas,r"$\delta$",nbins,f"Distr_delta-traj_{idx}.png")
plot(sigmas,r"$\sigma$",nbins,f"Distr_sigma-traj_{idx}.png")
fig, ax = plot(ls,r"$l$",nbins,f"Distr_l-traj_{idx}.png")
xxs = np.linspace(ls.min(),ls.max(),1000)
ax.plot(xxs,1/np.sqrt(2*np.pi*0.88**2)*np.exp(-0.5*(xxs-13)**2/0.88**2),color="black")
plot(phis,r"$\phi$",nbins,f"Distr_phi-traj_{idx}.png")
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
xxs = np.linspace(-50,50,1000)
ss = 13*2
print(ss)
ax.plot(xxs,1/np.sqrt(2*np.pi*ss**2)*np.exp(-0.5*(xxs-0)**2/ss**2),color="black")
ax.axvline(6.5)
ax.axvline(-6.5)
ax.axvline(13,ls="--")
ax.axvline(-13,ls="--")
plt.show()
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
ax.axvline(beta)
mb = np.mean(betas)
sdb = np.std(betas)
ax.fill_betweenx([deltas.min(),deltas.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 

height, _,_ = ax_histx.hist(betas, bins=bins, density=True,color="black")
ax_histx.axvline(np.mean(betas),ls="--")
ax_histx.axvline(beta)
ax_histx.fill_betweenx([0,height.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 

ax.axhline(np.mean(deltas),ls="--")
ax.axhline(delta)
md = np.mean(deltas)
sdd = np.std(deltas)
ax.fill_between([betas.min(),betas.max()], md - sdd, md + sdd, color=colors[0], alpha=0.15) 

height, _,_ = ax_histy.hist(deltas, bins=bins, density=True, orientation='horizontal',color="black")
ax_histy.axhline(np.mean(deltas),ls="--")
ax_histy.axhline(delta)
ax_histy.fill_between([0,height.max()],  md - sdd, md + sdd, color=colors[0], alpha=0.15) 

filename = f"Distr-traj_{idx}.png"
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
#v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
v,l,phi,Mu,Sigma,th0 = 5,13,0.9,0.0,13,1.0
param = np.array([v,l,phi,Mu,Sigma,th0 ])
icon = np.array([data[idx,0,0],data[idx,1,0],data[idx,2,0]])
t_fin = data[idx,3,-1]
Dt = data[idx,3,1]-data[idx,3,0]
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
    ax.plot(data[idx,0,:],data[idx,1,:],color="black")
    ax.set(xlabel=r"$x$",ylabel=r"$y$")
    ax.text(0.75,0.8,
            r"$\beta^*$={:.2f}".format(b_distr) +"\n" + r"$\delta^*$={:.2f}".format(d_distr),
            transform=ax.transAxes)

    sqd_pred = sqdispl(data_pred[xidxs],data_pred[yidxs])
    sqd = sqdispl(data[idx,0],data[idx,1])

    cros_pred = count_zero_crossings(data_pred[xidxs])
    cros =  count_zero_crossings(data[idx,0])

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

filename = f"Posterior_check-traj_{idx}.png"
fig.savefig(os.path.join(data_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
#%%

Ntraj_ic = 15
ks = np.array([0.6,0.12])
v,l,phi,Mu,Sigma,th0 = 5,13,0.9,0.0,13/2,1.0
param = np.array([v,l,phi,Mu,Sigma,th0 ])
data_pred = lib_model.multiple_traj(icon,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj_ic)
#%%
fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
for j in range(0,len(data_pred),3):
    #if j > 3*7: break
    ax.plot(data_pred[j],data_pred[(j+1)],alpha=0.5)
ax.axvline(0,color="black")
ax.axhline(0,color="black")
ax.set(xlabel=r"$x$",ylabel=r"$y$",xlim=[-70,70],ylim=[-10,800])