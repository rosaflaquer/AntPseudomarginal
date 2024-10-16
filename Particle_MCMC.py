#%%
import numpy as np
import os
#import matplotlib.pyplot as plt
import pandas as pd
from numba import get_num_threads
from lib_MCMC import *
from lib_model import model_step
import time as mtime


n_threads = get_num_threads()
print("Number of threads possible threads for this execution", n_threads)
space = 2 
numthreads = n_threads-space
print("Using", numthreads, "leaving ", space, "free")

dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
proj_path = os.path.split(dirname)[0] 
#plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
#prop_cycle = plt.rcParams['axes.prop_cycle']
#colors = prop_cycle.by_key()['color']
#%%
#1) Load observations
beta, delta, t_fin = 8, 0.3, 10
name = f"beta_{beta}-delta_{delta}-time_{t_fin}"
data_dir = os.path.join(proj_path,"Data","Synthetic",name)
data_file= f"Synthetic-{name}.dat"
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

ntraj_data = len(id_traj)
#plt.plot(data[0,0,:],data[0,1,:])
#plt.plot(data[1,0,:],data[1,1,:])
#plt.show(block=False)
#plt.close()
#%%

seed = 42
np.random.seed(seed)
R_trial = 100 #number of particles trial chains
M_trial = 5000 #number of MC steps trial chains
R = 100 #number of particles
M = 30000 #number of MC steps
ln_L0 = 0
C = 4 #number of chains
n_estim = 2

min_beta, max_beta = 4, 12
min_delta, max_delta = 0.005, 0.5
prior_pars = np.ones((2,2))
prior_pars[0][0],prior_pars[0][1] = min_beta, max_beta 
prior_pars[1][0],prior_pars[1][1] = min_delta, max_delta
init_params = [
    np.array([min_beta,min_delta]),
    np.array([min_beta,max_delta]),
    np.array([max_beta,min_delta]),
    np.array([max_beta,max_delta]),
] #Init the four chains at the edges of the parameter dsitr, to explore space. 

mean_kern = np.zeros(n_estim)
cov_kern = np.array([[prior_var(prior_pars[0])/100,0],
                     [0,prior_var(prior_pars[1])/100]])
print(cov_kern)
obs_li_param = np.array([0.2,0.2,0.045]) #From bbox histograms.

h = 0.01
sqh = np.sqrt(h)
v,l,phi,Mu,Sigma,th0 = 0.5,0.2,1.0,0.0,1.0,1.0
known_param = np.array([v,l,phi,Mu,Sigma,th0])

config_name = "cofig.dat"
with open(os.path.join(data_dir,config_name),"w") as f:
    f.write(f"numthreads = {numthreads} \n")
    f.write(f"seed = {seed} \n")
    f.write(f"R = {R_trial} \n")
    f.write(f"M = {M_trial} \n")
    f.write(f"R = {R} \n")
    f.write(f"M = {M} \n")
    f.write(f"ln_L0 = {ln_L0} \n")
    f.write(f"C = {C} \n")
    f.write(f"n_estim = {n_estim} \n")
    f.write(f"min_beta, max_beta = {min_beta}, {max_beta}\n")
    f.write(f"min_delta, max_delta = {min_delta}, {max_delta}\n")
    f.write(f"h = {h} \n")
    f.write(f"v,l,phi,Mu,Sigma,th0 = {v},{l},{phi},{Mu},{Sigma},{th0} \n")
    f.write(f"obs_li_param = {obs_li_param} \n")

#%%

def execute(init_params,cov_kern,log_file,chains_file,dfc,R,M,chains,ln_Ls):
    for i in range(C):
        param_0 = init_params[i]
        print("Chain",i,"init param", param_0)
        time_init = mtime.time()
        parameters_0,naccept,ln_L = ln_MCMC(M,n_estim,param_0,ln_L0,proposal_kernel_rv,prior_dist,prior_pars,mean_kern,cov_kern,ln_bootstrap,traj,R,model_step,log_obs_like,obs_li_param,h,sqh,known_param)
        time_fin = mtime.time()
        extime = time_fin - time_init
        print("Accepted", naccept, "ratio", naccept/M, "execution time", extime, "s", extime/60, "min")
        chains.append([parameters_0[:,0],parameters_0[:,1]])
        ln_Ls.append(ln_L)
        dfc[f"beta_{i}" ] = parameters_0[:,0]
        dfc[f"delta_{i}"] = parameters_0[:,1]
        dfc.to_csv(os.path.join(data_dir,chains_file),index=False)
        log_file.write(f"chain = {i} ################################################## \n")
        log_file.write(f"naccept = {naccept} \n")
        log_file.write(f"ln_L = {ln_L} \n")
        log_file.write(f"execution time = {extime} \n")

        #plt.plot(parameters_0[:,0])
        #plt.ylabel(r"$\beta$")
        #plt.axhline(beta,color="black")
        #plt.title(i)
        #plt.show(block=False)
        #plt.close()
        #plt.plot(parameters_0[:,1])
        #plt.ylabel(r"$\delta$")
        #plt.axhline(delta,color="black")
        #plt.title(i)
        #plt.show(block=False)
        #plt.close()


# %%

for idx in range(len(data[:2])):
    t_traj_ini = mtime.time()
    print(f"start with traj {idx}")
    traj = data[idx]
    chains_trial = []
    ln_Ls_trial = []
    dft = pd.DataFrame([])
    log_file = open(os.path.join(data_dir,f"log_trial_chains-traj_{idx}.dat"),"w")
    chains_file = f"Trial_chains-traj_{idx}.dat"
    execute(init_params,cov_kern,log_file,chains_file,dft,R_trial,M_trial,chains_trial,ln_Ls_trial)
    dft.to_csv(os.path.join(data_dir,chains_file),index=False)
    log_file.close()
    print("\n Done computing trial chains \n")

    par_complte = np.zeros(((M_trial+1)*C,n_estim))
    par_end = []
    for i in range(C):
        par_complte[i*(M_trial+1):(M_trial+1)*(i+1),0]  = chains_trial[i][0]
        par_complte[i*(M_trial+1):(M_trial+1)*(i+1),1]  = chains_trial[i][1]
        par_end.append([chains_trial[i][0][-1],chains_trial[i][1][-1],])

    sigma_opt = 2.38**2/n_estim*np.cov(par_complte,rowvar=False)
    print(sigma_opt,par_end)

    chains = []
    ln_Ls = []
    dfc = pd.DataFrame([])
    log_file = open(os.path.join(data_dir,f"log_chains-traj_{idx}.dat"),"w")
    chains_file = f"Chains-traj_{idx}.dat"
    execute(par_end,sigma_opt,log_file,chains_file,dfc,R,M,chains,ln_Ls)
    dfc.to_csv(os.path.join(data_dir,chains_file),index=False)
    log_file.close()
    hatR,ESS = convergence(chains,n_estim,C,M+1)
    log_file = open(os.path.join(data_dir,f"log_chains-traj_{idx}.dat"),"a")
    log_file.write("Convergence ########### \n")
    log_file.write(f"hatR = {hatR} \n")
    log_file.write(f"ESS = {ESS} \n")
    log_file.close()
    print(f"hatR = {hatR}, ESS = {ESS}")
    print("\n Done computing chains \n")
    
    t_traj_fin = mtime.time()
    extime = t_traj_fin-t_traj_ini
    print(f"Done with traj {idx} execution time {extime//60}:{extime%60} min \n \n \n")
print("Done")
#%%
