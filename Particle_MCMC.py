#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numba import get_num_threads, set_num_threads
from lib_MCMC import *
from lib_model import model_step
import time as mtime

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

with open(os.path.join(dirname,"config.par"),"r") as f:
    name = f.readline().split()[-1]
    idx = int(f.readline().split()[-1])
    numthreads = int(f.readline().split()[-1])
    seed = int(f.readline().split()[-1])
    R_trial = int(f.readline().split()[-1])
    M_trial = int(f.readline().split()[-1])
    R = int(f.readline().split()[-1])
    M = int(f.readline().split()[-1])
    ln_L0 = int(f.readline().split()[-1])
    C = int(f.readline().split()[-1])
    n_estim = int(f.readline().split()[-1])
    h = float(f.readline().split()[-1])
    first_beta = float(f.readline().split()[-1])
    second_beta = float(f.readline().split()[-1])
    distr_beta = int(f.readline().split()[-1])
    first_delta = float(f.readline().split()[-1])
    second_delta = float(f.readline().split()[-1])
    distr_delta = int(f.readline().split()[-1])
    v = float(f.readline().split()[-1])
    first_l = float(f.readline().split()[-1])
    second_l = float(f.readline().split()[-1])
    distr_l = int(f.readline().split()[-1])
    first_phi = float(f.readline().split()[-1])
    second_phi = float(f.readline().split()[-1])
    distr_phi = int(f.readline().split()[-1])
    Mu = float(f.readline().split()[-1])
    first_sigma = float(f.readline().split()[-1])
    second_sigma = float(f.readline().split()[-1])
    distr_sigma = int(f.readline().split()[-1])
    th0 = float(f.readline().split()[-1])
    obs_li_param_x = float(f.readline().split()[-1])
    obs_li_param_y = float(f.readline().split()[-1])
    obs_li_param_th = float(f.readline().split()[-1])
    frac_var_obs = float(f.readline().split()[-1])

#%%

number_of_available = get_num_threads() 
set_num_threads(numthreads)
print("Using", numthreads, "leaving ", number_of_available - numthreads, "free")


#%%
#1) Load observations

data_dir = os.path.join(proj_path,"Data","Synthetic",name)
data_file= f"Synthetic-{name}.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))
id_traj = datadf["id_traj"].unique()
Ntrajs = len(id_traj)
len_trajs = len(datadf[datadf["id_traj"]==id_traj[0]])
data = np.zeros((Ntrajs,4,len_trajs))
sdat = 0.05
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

np.random.seed(seed)
sqh = np.sqrt(h)
known_param = np.array([v,Mu,th0])
prior_pars = np.ones((n_estim,3))

prior_pars[0][0],prior_pars[0][1],prior_pars[0][2] = first_beta, second_beta, distr_beta
prior_pars[1][0],prior_pars[1][1],prior_pars[1][2] = first_delta, second_delta, distr_delta
prior_pars[2][0],prior_pars[2][1],prior_pars[2][2] = first_sigma, second_sigma, distr_sigma
prior_pars[3][0],prior_pars[3][1],prior_pars[3][2] = first_l,second_l, distr_l
prior_pars[4][0],prior_pars[4][1],prior_pars[4][2] = first_phi,second_phi, distr_phi
if distr_sigma == 0: Sigma = second_sigma - first_sigma
elif distr_sigma == 1: Sigma = first_sigma
if distr_l == 0: l = second_l - first_l
elif distr_l == 1: l = first_l
if distr_phi == 0: phi = second_phi - first_phi
elif distr_phi == 1: phi = first_phi

init_params = [
    np.array([first_beta,first_delta,Sigma,l,phi]),
    np.array([first_beta,second_delta,Sigma,l,phi]),
    np.array([second_beta,first_delta,Sigma,l,phi]),
    np.array([second_beta,second_delta,Sigma,l,phi]),
] #Init the four chains at the edges of the parameter dsitr, to explore space. 


mean_kern = np.zeros(n_estim)
cov_kern = np.zeros((n_estim,n_estim))
for i in range(n_estim):
    cov_kern[i][i] = prior_var(prior_pars[i])/frac_var_obs
print(cov_kern)
obs_li_param = np.array([obs_li_param_x,obs_li_param_y,obs_li_param_th*np.pi/180])

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
    for i in range(C):
        f.write(f"param {i} = {init_params[i]} \n")
    f.write(f"h = {h} \n")
    f.write(f"v,l,phi,Mu,Sigma,th0 = {v},{l},{phi},{Mu},{Sigma},{th0} \n")
    f.write(f"obs_li_param = {obs_li_param} \n")
    f.write(f"cov_kern = {cov_kern} \n")

#%%


def execute(init_params,cov_kern,ln_Ls,R,M,C,log_file_name,chains_file,data_dir):
    chains = np.zeros((C,n_estim,M+1))
    dfc = pd.DataFrame([])
    log_file = open(os.path.join(data_dir,log_file_name),"w")

    for i in range(C):
        param_0 = init_params[i]
        print("\n Chain",i,"init param", param_0,"####################"*10,"\n")
        time_init = mtime.time()
        parameters_0,naccept,ln_L = ln_MCMC(M,n_estim,param_0,ln_Ls[i],proposal_kernel_rv,prior_dist,prior_pars,mean_kern,cov_kern,ln_bootstrap,traj,R,model_step,log_obs_like,obs_li_param,h,sqh,known_param)
        time_fin = mtime.time()
        extime = time_fin - time_init
        print("execution time", extime, "s", extime/60, "min")
        for j in range(n_estim):
            chains[i][j]= parameters_0[:,j]
            dfc[f"par{j}_{i}"] = parameters_0[:,j]
        ln_Ls[i] = ln_L[-1]
        dfc[f"ln_L_{i}"] = naccept
        dfc[f"accep_{i}"] = ln_L
        dfc.to_csv(os.path.join(data_dir,chains_file),index=False)
        log_file.write(f"chain = {i} \n")
        log_file.write(f"execution time = {extime} \n")
    hatR,ESS = convergence(chains,n_estim,C,M+1)
    log_file.write("Convergence ########### \n")
    log_file.write(f"hatR = {hatR} \n")
    log_file.write(f"ESS = {ESS} \n")
    log_file.close()
    print(f"hatR = {hatR}, ESS = {ESS}")
    return chains,ln_Ls
    

#%%

t_traj_ini = mtime.time()
print(f"start with traj {idx}")
traj = data[idx]
ln_Ls = np.ones(C)*ln_L0

log_file_name = f"log_trial_chains-traj_{idx}.dat"
chains_file = f"Trial_chains-traj_{idx}.dat"
chains_trial,ln_Ls_trial = execute(init_params,cov_kern,ln_Ls,R_trial,M_trial,C,log_file_name,chains_file,data_dir)
print("\n Done computing trial chains \n")

#compute optimal sigma
par_complte = np.zeros(((M_trial+1)*C,n_estim))
par_end = []
for i in range(C):
    last_par = []
    for j in range(n_estim):
        par_complte[i*(M_trial+1):(M_trial+1)*(i+1),j]  = chains_trial[i][j]
        last_par.append(chains_trial[i][j][-1])
    par_end.append(last_par)
sigma_opt = 2.38**2/n_estim*np.cov(par_complte,rowvar=False)
print(sigma_opt,par_end)


ln_Ls = ln_Ls_trial
log_file_name = f"log_chains-traj_{idx}.dat"
chains_file = f"Chains-traj_{idx}.dat"
chains_trial,ln_Ls_trial = execute(par_end,sigma_opt,ln_Ls,R,M,C,log_file_name,chains_file,data_dir)
print("\n Done computing chains \n")

t_traj_fin = mtime.time()
extime = t_traj_fin-t_traj_ini
print(f"Done with traj {idx} execution time {extime//60}:{extime%60} min \n \n \n")

#%%
