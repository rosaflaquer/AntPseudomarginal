#%%
"""
Cite numba-scipy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numba import get_num_threads, njit
from lib_MCMC import *
from lib_model import model_step
from numba_stats import norm,uniform
import scipy as sp

n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 6 
numthreads = n_threads-space
print("Using", numthreads, "leaving ", space, "free")

dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#%%
"""
Initial definitions
"""

#1) Load observations
data_dir = os.path.join(proj_path,"Data","Synthetic")
data_file = os.listdir(data_dir)[0]
datadf = pd.read_csv(os.path.join(data_dir,data_file))
id_traj = datadf["id_traj"].unique()
Ntrajs = len(id_traj)
len_trajs = len(datadf[datadf["id_traj"]==id_traj[0]])
data = np.zeros((Ntrajs,4,len_trajs))
for i,id in enumerate(id_traj):
    day_traj = datadf[datadf["id_traj"]==id]
    data[i][0] = day_traj["x"].values + np.random.normal(0,0.00001,size=len(day_traj["x"]))
    data[i][1] = day_traj["y"].values + np.random.normal(0,0.00001,size=len(day_traj["y"]))
    data[i][2] = day_traj["theta"].values + np.random.normal(0,0.00001,size=len(day_traj["theta"]))
    data[i][3] = day_traj["Time"].values

#%%
#2) Initial conditions. If trajectory by trajectory the distr are directly the IC of the given traj.

#prob, bins and cumulative distr of the initial conditons from the sampled data.
px,bx,cx    = histo_from_data(data[:,0,0]) 
py,by,cy    = histo_from_data(data[:,1,0])
pth,bth,cth = histo_from_data(data[:,2,0])

#%%
#3) Prior distribution for the parameters 

min_beta, max_beta = 3.95, 4.05
min_delta, max_delta = 0.095, 0.105
prior_pars = np.ones((2,2))
prior_pars[0][0],prior_pars[0][1] = min_beta, max_beta 
prior_pars[1][0],prior_pars[1][1] = min_delta, max_delta
param_0 = prior_rvs(prior_pars)
#%% 
# 4-5) Define the number of particles to use and the MC steps
R = 5 #number of particles
M = 10 #number of MC steps

#%%
# 6) initial value for the likelihood. 
# TODO: Use the log likelihood (once implemented)

L0 = 1

#%%
# 7) Definitions of the proposals and obs likely
n_estim = 2
mean_kern = np.zeros(n_estim)
cov_kern = np.array([[0.01,0],[0,0.01]]) #TODO change that to sth that makes sense
#obs_li_param = np.array([0.25,0.25,0.05]) #np.ones(3)*0.25 #TODO change that to sth that makes sense
obs_li_param = np.ones(3) #TODO change that to sth that makes sense

#%%

#simu definitions

h = 0.01
sqh = np.sqrt(h)
v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
known_param = np.array([v,l,phi,Mu,Sigma,th0])

# %%
for traj in data[:1]:
    parameters,Ls = MCMC(M,n_estim,param_0,L0,proposal_kernel_rv,prior_dist,prior_pars,mean_kern,cov_kern,bootstrap,traj,R,model_step,obs_like,obs_li_param,h,sqh,known_param)
    print(parameters,Ls)
# %%
