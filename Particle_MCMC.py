#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numba import get_num_threads
from lib_MCMC import *
from lib_model import model_step

n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 3 
numthreads = n_threads-space
print("Using", numthreads, "leaving ", space, "free")

dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#%%
#1) Load observations
data_dir = os.path.join(proj_path,"Data","Synthetic")
data_file = os.listdir(data_dir)[0]
data_file= "Synthetic-beta_4-delta_0.3-time_10.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))
id_traj = datadf["id_traj"].unique()
Ntrajs = len(id_traj)
len_trajs = len(datadf[datadf["id_traj"]==id_traj[0]])
data = np.zeros((Ntrajs,4,len_trajs))
sdat = 0.005
for i,id in enumerate(id_traj):
    day_traj = datadf[datadf["id_traj"]==id]
    data[i][0] = day_traj["x"].values     + np.random.normal(0,sdat,size=len(day_traj["x"]))
    data[i][1] = day_traj["y"].values     + np.random.normal(0,sdat,size=len(day_traj["y"]))
    data[i][2] = day_traj["theta"].values + np.random.normal(0,sdat,size=len(day_traj["theta"]))
    data[i][3] = day_traj["Time"].values
#%%

plt.plot(data[1,0,:],data[1,1,:])

#%%
#2) Initial conditions. If trajectory by trajectory the distr are directly the IC of the given traj.

#prob, bins and cumulative distr of the initial conditons from the sampled data.
px,bx,cx    = histo_from_data(data[:,0,0]) 
py,by,cy    = histo_from_data(data[:,1,0])
pth,bth,cth = histo_from_data(data[:,2,0])

#%%

R = 100 #number of particles
M = 5000 #number of MC steps
ln_L0 = 0
C = 4 #number of chains
n_estim = 2

min_beta, max_beta = 2, 6
min_delta, max_delta = 0.2, 0.5
prior_pars = np.ones((2,2))
prior_pars[0][0],prior_pars[0][1] = min_beta, max_beta 
prior_pars[1][0],prior_pars[1][1] = min_delta, max_delta
init_params = [prior_rvs(prior_pars) for i in range(C)]
init_params = [
    np.array([min_beta,min_delta]),
    np.array([min_beta,max_delta]),
    np.array([max_beta,min_delta]),
    np.array([max_beta,max_delta]),
]

mean_kern = np.zeros(n_estim)
cov_kern = np.array([[0.1,0],[0,0.05]]) #TODO change that to sth that makes sense
cov_kern = np.array([[prior_var(prior_pars[0])/100,0],
                     [0,prior_var(prior_pars[1])]])
print(cov_kern)
obs_li_param = np.array([0.2,0.2,0.045])
#%%
#simu definitions
h = 0.01
sqh = np.sqrt(h)
v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
known_param = np.array([v,l,phi,Mu,Sigma,th0])

# %%
chains = []
for i in range(C):
    for traj in data[1:2]:
        param_0 = init_params[i]
        print(param_0)
        parameters_0,naccept,ln_L = ln_MCMC(M,n_estim,param_0,ln_L0,proposal_kernel_rv,prior_dist,prior_pars,mean_kern,cov_kern,ln_bootstrap,traj,R,model_step,log_obs_like,obs_li_param,h,sqh,known_param)
        print("Accepted", naccept, "ratio", naccept/M )
    chains.append(parameters_0)

    plt.plot(parameters_0[:,0])
    plt.ylabel(r"$\beta$")
    plt.axhline(4,color="black")
    plt.show()

    plt.plot(parameters_0[:,1])
    plt.ylabel(r"$\delta$")
    plt.axhline(0.3,color="black")
    plt.show()

# %%

parameters_0 = np.zeros((len(chains[0])*len(chains),2))
for i in range(C):
    parameters_0[i*len(chains[0]):len(chains[0])*(i+1),0]  = chains[i][:,0]
    parameters_0[i*len(chains[0]):len(chains[0])*(i+1),1]  = chains[i][:,1]

sigma_opt = 2.38**2/n_estim*np.cov(parameters_0,rowvar=False)
obs_li_param = np.array([0.2,0.2,0.045])
param_0_opt = parameters_0[-1]
print(param_0_opt)
print(sigma_opt)
#%%

# 4-5) Define the number of particles to use and the MC steps
R = 100 #number of particles
M = 25000 #number of MC steps
C = 4 #number of chains

#%%
chains_conv = []

for i in range(C):
    param_0_opt = chains[i][-1]
    print(i,param_0_opt)
    for traj in data[1:2]:
        parameters, naccept, ln_L = ln_MCMC(M,n_estim,param_0_opt,ln_L0,proposal_kernel_rv,prior_dist,prior_pars,mean_kern,sigma_opt,ln_bootstrap,traj,R,model_step,log_obs_like,obs_li_param,h,sqh,known_param)
        print("Accepted", naccept, "ratio", naccept/M )

    chains_conv.append(parameters)
    nbins = 50
    minimums = np.int32(np.array([0,0.1,0.2,0.5,0.75])*M)
    for i,minim in enumerate(minimums):
        n, bins = np.histogram(parameters[minim:,0],bins=nbins,density=True)
        bins = (bins[1:]+bins[:-1])*0.5
        plt.plot(bins,n,label=minim)
        #plt.hist(parameters[minim:,0],bins=nbins,density=True,label=minim)
        plt.axvline(param_0_opt[0],color="black",ls="--")
        plt.axvline(4,color="black")
        plt.axvline(np.mean(parameters[minim:,0]),color=colors[i]) #,color="grey")
        plt.xlabel(r"$\beta$")
        plt.legend()
    plt.show()
    for i,minim in enumerate(minimums):
        n, bins = np.histogram(parameters[minim:,1],bins=nbins,density=True)
        bins = (bins[1:]+bins[:-1])*0.5
        plt.plot(bins,n,label=minim)
        #plt.hist(parameters[minim:,0],bins=nbins,density=True,label=minim)
        plt.axvline(param_0_opt[1],color="black",ls="--")
        plt.axvline(0.3,color="black")
        plt.axvline(np.mean(parameters[minim:,1]),color=colors[i])  #,color="grey")
        plt.xlabel(r"$\delta$")
        plt.legend()
    plt.show()

    plt.plot(parameters[:,0])
    plt.ylabel(r"$\beta$")
    plt.axhline(4,color="black")
    plt.show()

    plt.plot(parameters[:,1])
    plt.ylabel(r"$\delta$")
    plt.axhline(0.3,color="black")
    plt.show()
#%%
out_path = os.path.join(proj_path,"Data","Simulations")
dfc = pd.DataFrame([])
for i in range(C):
    dfc[f"beta_{i}" ] = chains[i][:,0]
    dfc[f"delta_{i}"] = chains[i][:,1]
dfc.to_csv(os.path.join(out_path,"Trial_chains.dat"),index=False)
dfc = pd.DataFrame([])
for i in range(C):
    dfc[f"beta_{i}" ] = chains_conv[i][:,0]
    dfc[f"delta_{i}"] = chains_conv[i][:,1]
dfc.to_csv(os.path.join(out_path,"Chains.dat"),index=False)

#%%

#%%
comp = 0
MM = M+1
combined = np.zeros((n_estim,C*MM))
for i in range(C):
    for p in range(n_estim):
        print(i,p,len(chains_conv[i][:,p]))
        print(len(combined[p,i*MM:MM*(i+1)]))
        combined[p,i*MM:MM*(i+1)] = chains_conv[i][:,p]
    comp += len(chains_conv[i][:,p])

#%%
#%%
