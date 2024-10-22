#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lib_model
from numba import get_num_threads

n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 6 
numthreads = n_threads-space
print("Using", numthreads, "leaving ", space, "free")
#%%

#directory definition
dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
path = os.path.split(dirname)[0] #crop before scripts
plt.style.use(os.path.join(os.path.split(path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def plot(axs,i,j,idx,idy,labelx,labely,xlim,ylim):
    for k in range(Ntraj):
        if type(idx) == str:
            axs[i][j].plot(times,timeevols[idy[k]])
        else:
            axs[i][j].plot(timeevols[idx[k]],timeevols[idy[k]])
    axs[i][j].set(xlim=xlim,ylim=ylim)
    if i == 0: axs[i][j].set_title(r"$\delta$={}".format(delta))
    if i == len(betas)-1 and j!=0:
        axs[i][j].set(yticklabels=[],xlabel=labelx)
    elif i == len(betas)-1 and j==0:
        axs[i][j].set(xlabel=labelx,ylabel=r"$\beta$={}".format(beta) +"\n"+ labely)
    elif j==0 and i != len(betas)-1:
        axs[i][j].set(xticklabels=[],ylabel=r"$\beta$={}".format(beta) +"\n"+ labely)
    else:
        axs[i][j].set(xticklabels=[],yticklabels=[])

#%%
#time evolution

#inital conditions
x0 = 0.0
y0 = 0.0
theta0 = np.pi/2-0.5
ci = np.array([x0,y0,theta0])
v,l,phi,Mu,Sigma,th0 = 5,13,0.9,0.0,13/2,1.0
param = np.array([v,l,phi,Mu,Sigma,th0 ])


#parameters of the simulation
t_fin = 150 
dwr = 1 
h = 0.01 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
Ntraj = 15
Ntraj_ic = Ntraj

xindx = np.arange(0,Ntraj*len(ci),len(ci))
yindx = np.arange(1,Ntraj*len(ci),len(ci))
thindx= np.arange(2,Ntraj*len(ci),len(ci))
betas = [0.1,0.25,0.5,1,2]
deltas = [0.001,0.01,0.05,0.1,0.3]
times = np.arange(0,t_fin,dwr)


fig1, axs1 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
fig2, axs2 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
fig3, axs3 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
fig4, axs4 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
fig5, axs5 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
fig1.subplots_adjust(wspace=0, hspace=0)
fig2.subplots_adjust(wspace=0, hspace=0)
fig3.subplots_adjust(wspace=0, hspace=0)
fig4.subplots_adjust(wspace=0, hspace=0)
fig5.subplots_adjust(wspace=0, hspace=0)
xmin, xmax = -40,40
ymin, ymax = -1,700
for i,beta in enumerate(betas):
    for j,delta in enumerate(deltas):
        ks = np.array([beta,delta])
        ci = np.array([x0,y0,theta0])
        print(f"beta = {beta}, delta = {delta}")
        timeevols = lib_model.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj_ic)
        #XY PLOT
        plot(axs1,i,j,xindx,yindx,r"$x$",r"$y$",[xmin,xmax],[ymin,ymax])   

        #XTHETA PLOT
        plot(axs2,i,j,xindx,thindx,r"$x$",r"$\theta$",[xmin,xmax],[-0.1,2*np.pi + 0.1])   

        #XT PLOT
        plot(axs3,i,j,"tt",xindx,r"$t$",r"$x$",[0,times[-1]],[xmin,xmax])

        #YT PLOT
        plot(axs4,i,j,"tt",yindx,r"$t$",r"$y$",[0,times[-1]],[ymin,ymax])

        #YT PLOT
        plot(axs5,i,j,"tt",thindx,r"$t$",r"$\theta$",[0,times[-1]],[-0.1,2*np.pi + 0.1])  


#%%
fig1.savefig(os.path.join(path,f"XY_plot.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig2.savefig(os.path.join(path,f"XTHETA_plot.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig3.savefig(os.path.join(path,f"Xt_plot.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig4.savefig(os.path.join(path,f"Yt_plot.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig5.savefig(os.path.join(path,f"THETAt_plot.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")