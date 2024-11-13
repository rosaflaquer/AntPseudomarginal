#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numba import get_num_threads,set_num_threads
import lib_model as lib
import time


n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 4 #n_threads - 4
numthreads = n_threads-space
set_num_threads(numthreads)
print("Using", numthreads, "leaving ", space, "free")
#%%
save = False


dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
del(dirname)
folders = ["Data","Simulations"]
for folder in folders:
    path = os.path.join(path,folder)
    if not(os.path.exists(path)): os.mkdir(path)


def plot(axs,i,j,idx,idy,labelx,labely,xlim,ylim,Ntraj,xticks,yticks):
    for k in range(Ntraj):
        if type(idx) == str:
            axs[i][j].plot(times,timeevols[idy[k]])
        else:
            axs[i][j].plot(timeevols[idx[k]],timeevols[idy[k]])
            axs[i][j].axvline(0,color="black")
    axs[i][j].set(xlim=xlim,ylim=ylim)
    if i == 0: axs[i][j].set_title(r"$\delta$={}".format(delta))
    if i == len(betas)-1 and j!=0:
        axs[i][j].set(yticklabels=[],xlabel=labelx,xticks=xticks)
    elif i == len(betas)-1 and j==0:
        axs[i][j].set(xlabel=labelx,ylabel=r"$\beta$={}".format(beta) +"\n"+ labely , yticks=yticks, xticks=xticks)
    elif j==0 and i != len(betas)-1:
        axs[i][j].set(xticklabels=[],ylabel=r"$\beta$={}".format(beta) +"\n"+ labely, yticks=yticks)
    else:
        axs[i][j].set(xticklabels=[],yticklabels=[])

#%%
#time evolution
x0 = 0.0
y0 = 0.0
theta0 = np.pi/2 - 0.1
ci = np.array([x0,y0,theta0])


#parameters of the simulation
tfin = 50
h = 0.001
Nt = int(tfin/h)
dw = 0.01
iwr = int(dw/h)
Ntraj = 20

#parameters of the model
v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
param = np.array([v,l,phi,Mu,Sigma,th0])#This is what we want to inffer!
xindx = np.arange(0,Ntraj*len(ci),len(ci))
yindx = np.arange(1,Ntraj*len(ci),len(ci))
thindx= np.arange(2,Ntraj*len(ci),len(ci))
betas = [0.5,1,2,4,8,16,32]
deltas = [0.01,0.1,0.3,0.6]
times = np.arange(0,tfin,dw)


fig1, axs1 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
#fig2, axs2 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
#fig3, axs3 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
#fig4, axs4 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
#fig5, axs5 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
#fig6, axs6 = plt.subplots(ncols=len(deltas),nrows=len(betas),figsize=(2.5*len(betas),2.5*len(deltas)))
fig1.subplots_adjust(wspace=0, hspace=0)
#fig2.subplots_adjust(wspace=0, hspace=0)
#fig3.subplots_adjust(wspace=0, hspace=0)
#fig4.subplots_adjust(wspace=0, hspace=0)
#fig5.subplots_adjust(wspace=0, hspace=0)
xmin, xmax = -2.5,2.5
ymin, ymax = -1,25
for i,beta in enumerate(betas):
    for j,delta in enumerate(deltas):
        param = np.array([v,l,phi,Mu,Sigma,th0]) 
        ks = np.array([beta, delta]) 
        ci = np.array([x0,y0,theta0])
        print(f"beta = {beta}, delta = {delta}")
        timeevols = lib.multiple_traj(ci,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj)
       
        #XY PLOT
        xticks = np.array([-1.5,0,1.5])
        yticks = np.array([5,12.5,20])
        plot(axs1,i,j,xindx,yindx,r"$x$",r"$y$",[xmin,xmax],[ymin,ymax],Ntraj,xticks,yticks)

        ##XTHETA PLOT
        #plot(axs2,i,j,xindx,thindx,r"$x$",r"$\theta$",[xmin,xmax],[-0.1,2*np.pi + 0.1],Ntraj)
#
        ##XT PLOT
        #plot(axs3,i,j,"tt",xindx,r"$t$",r"$x$",[0,times[-1]],[xmin,xmax],Ntraj)
#
        ##YT PLOT
        #plot(axs4,i,j,"tt",yindx,r"$t$",r"$y$",[0,times[-1]],[ymin,ymax],Ntraj)
#
        ##YT PLOT
        #plot(axs5,i,j,"tt",thindx,r"$t$",r"$\theta$",[0,times[-1]],[-0.1,2*np.pi + 0.1],Ntraj)  


#%%
fig1.savefig(os.path.join(path,f"XY_plot_pi_2.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
#fig2.savefig(os.path.join(path,f"XTHETA_plot.png"),format="png",
#            facecolor="w",edgecolor="w",bbox_inches="tight")
#fig3.savefig(os.path.join(path,f"Xt_plot.png"),format="png",
#            facecolor="w",edgecolor="w",bbox_inches="tight")
#fig4.savefig(os.path.join(path,f"Yt_plot.png"),format="png",
#            facecolor="w",edgecolor="w",bbox_inches="tight")
#fig5.savefig(os.path.join(path,f"THETAt_plot.png"),format="png",
#            facecolor="w",edgecolor="w",bbox_inches="tight")
# %%
