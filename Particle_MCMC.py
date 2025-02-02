#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numba import get_num_threads, set_num_threads
from lib_MCMC import *
from lib_model_extended import model_step
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
    first_gamma = float(f.readline().split()[-1])
    second_gamma = float(f.readline().split()[-1])
    distr_gamma = int(f.readline().split()[-1])
    neq = int(f.readline().split()[-1])
    neq_dat = int(f.readline().split()[-1])

#%%


set_num_threads(numthreads)


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

#%%
ntraj_data = len(id_traj)
plt.plot(data[idx,0,:],data[idx,1,:])
plt.show(block=False)
print(data[idx,:,0])
plt.legend()
plt.close()
#%%

np.random.seed(seed)
sqh = np.sqrt(h)
prior_pars = np.ones((n_estim,3))

prior_pars[0][0],prior_pars[0][1],prior_pars[0][2] = first_beta, second_beta, distr_beta
prior_pars[1][0],prior_pars[1][1],prior_pars[1][2] = first_delta, second_delta, distr_delta
#prior_pars[2][0],prior_pars[2][1],prior_pars[2][2] = first_sigma, second_sigma, distr_sigma
#prior_pars[3][0],prior_pars[3][1],prior_pars[3][2] = first_l,second_l, distr_l
#prior_pars[4][0],prior_pars[4][1],prior_pars[4][2] = first_phi,second_phi, distr_phi
#prior_pars[5][0],prior_pars[5][1],prior_pars[5][2] = first_gamma,second_gamma, distr_gamma
#prior_pars[2][0],prior_pars[2][1],prior_pars[2][2] = first_gamma,second_gamma, distr_gamma
if distr_sigma == 0: Sigma = (second_sigma - first_sigma)/2 + first_sigma
elif distr_sigma == 1: Sigma = first_sigma
if distr_l == 0: l = (second_l - first_l)/2 + first_l
elif distr_l == 1: l = first_l
if distr_phi == 0: phi = (second_phi - first_phi)/2 + first_phi
elif distr_phi == 1: phi = first_phi
if distr_gamma == 0:   gamma = (second_gamma - first_gamma)/2 + first_gamma
elif distr_gamma == 1: gamma = first_gamma

known_param = np.array([v,Mu,th0,Sigma,l,phi,gamma])
#known_param = np.array([v,Mu,th0])

init_params = [
    np.array([(second_beta - first_beta)/2*(1+0.25) + first_beta,(second_delta - first_delta)/2*(1+0.25) + first_delta ]), #gamma ]),
    np.array([(second_beta - first_beta)/2*(1+0.25) + first_beta,(second_delta - first_delta)/2*(1-0.25) + first_delta ]), #gamma ]),
    np.array([(second_beta - first_beta)/2*(1-0.25) + first_beta,(second_delta - first_delta)/2*(1+0.25) + first_delta ]), #gamma ]),
    np.array([(second_beta - first_beta)/2*(1-0.25) + first_beta,(second_delta - first_delta)/2*(1-0.25) + first_delta ]), #gamma ]),
] #Init the four chains at the edges of the parameter dsitr, to explore space. 


#init_params = [
#    np.array([first_beta+0.001, first_delta+0.001 ]),
#    np.array([first_beta+0.001, second_delta-0.001]),
#    np.array([second_beta-0.001,first_delta+0.001 ]),
#    np.array([second_beta-0.001,second_delta-0.001]),
#] #Init the four chains at the edges of the parameter dsitr, to explore space. 

print(init_params)

#init_params = [
#    np.array([first_beta+0.001, first_delta+0.001 , gamma]),
#    np.array([first_beta+0.001, second_delta-0.001, gamma]),
#    np.array([second_beta-0.001,first_delta+0.001 , gamma]),
#    np.array([second_beta-0.001,second_delta-0.001, gamma ]),
#] #Init the four chains at the edges of the parameter dsitr, to explore space. 


#init_params = [
#    np.array([first_beta+0.001,first_delta+0.001,Sigma,l,phi  ,gamma]),
#    np.array([first_beta+0.001,second_delta-0.001,Sigma,l,phi ,gamma]),
#    np.array([second_beta-0.001,first_delta+0.001,Sigma,l,phi ,gamma]),
#    np.array([second_beta-0.001,second_delta-0.001,Sigma,l,phi,gamma ]),
#] #Init the four chains at the edges of the parameter dsitr, to explore space. 

mean_kern = np.zeros(n_estim)
cov_kern = np.zeros((n_estim,n_estim))
for i in range(n_estim):
    cov_kern[i][i] = prior_var(prior_pars[i])/frac_var_obs
cov_kern = np.sqrt(cov_kern)
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
    f.write(f"v,l,phi,Mu,Sigma,th0 = {v},{l},{phi},{Mu},{Sigma},{th0},{gamma} \n")
    f.write(f"obs_li_param = {obs_li_param} \n")
    f.write(f"cov_kern = {cov_kern} \n")



#%%

print(prior_dist(init_params[0],prior_pars))

#%%

t_traj_ini = mtime.time()
print(f"start with traj {idx}")
traj = data[idx]
ln_Ls = np.ones(C)*log_obs_like(traj[:neq_dat,0],traj[:neq_dat,0],neq_dat,obs_li_param)
print(ln_Ls)

#%%

log_file_name = f"log_trial_chains-traj_{idx}.dat"
chains_file = f"Trial_chains-traj_{idx}.dat"
chains_trial,ln_Ls_trial,accepted_trial = execute(init_params,mean_kern,cov_kern,ln_Ls,R_trial,M_trial,C,n_estim,prior_pars,obs_li_param,known_param,log_file_name,chains_file,data_dir,traj,model_step,h,sqh,neq,neq_dat)
print("\n Done computing trial chains \n", accepted_trial)

#compute optimal sigma
par_complte = np.zeros(((M_trial+1)*C,n_estim))
par_end = []
for i in range(C):
    last_par = []
    if accepted_trial[i] < 0.1:
        continue
    for j in range(n_estim):
        par_complte[i*(M_trial+1):(M_trial+1)*(i+1),j]  = chains_trial[i][j]
        last_par.append(chains_trial[i][j][-1])
    par_end.append(last_par)
sigma_opt = 2.38**2/n_estim*np.cov(par_complte,rowvar=False)
#sigma_opt = np.sqrt(sigma_opt)
print(sigma_opt,cov_kern)
if np.isnan(sigma_opt).any() or np.any(np.diagonal(sigma_opt) == 0):
    sigma_opt = cov_kern
    print("sigma_opt is nan or accepted is low, using cov_kern instead")
if len(par_end) == 0:
    print("Bad range")
else: 
    out_range = False
    present = len(par_end)-1
    while len(par_end) < C:
        if present == 0: par_end.append(par_end[0])
        else: par_end.append(par_end[np.random.randint(0,present)])
    print(par_end)
    ln_Ls = ln_Ls_trial
    log_file_name = f"log_chains-traj_{idx}.dat"
    chains_file = f"Chains-traj_{idx}.dat"
    chains,ln_Ls,accepted = execute(par_end,mean_kern,sigma_opt,ln_Ls,R,M,C,n_estim,prior_pars,obs_li_param,known_param,log_file_name,chains_file,data_dir,traj,model_step,h,sqh,neq,neq_dat)
    print("\n Done computing chains \n", accepted)

    t_traj_fin = mtime.time()
    extime = t_traj_fin-t_traj_ini
    print(f"Done with traj {idx} execution time {extime//60}:{extime%60} min \n \n \n")

#%%
