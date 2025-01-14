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

with open(os.path.join(dirname,"config_data.par"),"r") as f:
    name = f.readline().split()[-1]
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
    id_trajs_file = f.readline().split()[-1]

with open(os.path.join(dirname,id_trajs_file),"r") as f:
    id_list = f.readlines()

number_of_available = get_num_threads() 
set_num_threads(numthreads)
print("Using", numthreads, "leaving ", number_of_available - numthreads, "free")

#%%
data_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = name
datadf = pd.read_csv(os.path.join(data_dir,data_file))

np.random.seed(seed)
sqh = np.sqrt(h)


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
    np.array([first_beta ,first_delta ,Sigma,l,phi]),
    np.array([first_beta ,second_delta,Sigma,l,phi]),
    np.array([second_beta,first_delta ,Sigma,l,phi]),
    np.array([second_beta,second_delta,Sigma,l,phi]),
] #Init the four chains at the edges of the parameter dsitr, to explore space. 


mean_kern = np.zeros(n_estim)
cov_kern = np.zeros((n_estim,n_estim))
for i in range(n_estim):
    cov_kern[i][i] = prior_var(prior_pars[i])/frac_var_obs
cov_kern = np.sqrt(cov_kern)
obs_li_param = np.array([obs_li_param_x,obs_li_param_y,obs_li_param_th*np.pi/180])


for item in id_list:
    if item[0] == "#": continue
    id_traj = item.strip("\n")
    id_folder = id_traj
    is_segment = id_folder.find("0_s")
    id_folder = id_folder[:is_segment+2]
    if is_segment > 0: out_dir_list = ["Data","Fits","NoPause",f"Traj_{id_folder}"]
    else: out_dir_list = ["Data","Fits",f"Traj_{id_traj}"]
    out_dir = proj_path
    for directory in out_dir_list:
        out_dir = os.path.join(out_dir,directory)
        if not(os.path.exists(out_dir)): os.mkdir(out_dir)
    len_trajs = len(datadf[datadf["id_traj"]==id_traj])
    data = np.zeros((4,len_trajs))
    day_traj = datadf[datadf["id_traj"]==id_traj]
    data[0] = day_traj["x"].values     
    data[1] = day_traj["y"].values     
    data[2] = day_traj["theta"].values 
    data[3] = day_traj["Time"].values
    vs = day_traj["$|v|$"].values
    v = day_traj["$|v|$"].mean()
    known_param = np.array([v,Mu,th0])


    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
    cm = ax.scatter(data[0,:],data[1,:],c=vs)    
    ax.set(xlabel=r"$x$",ylabel=r"$y$",title=id_traj)
    ax.axvline(0,color="black")
    plt.colorbar(cm,ax=ax)
    filename = f"Traj_{id_traj}.png"
    fig.savefig(os.path.join(out_dir,filename),format="png",
                facecolor="w",edgecolor="w",bbox_inches="tight")

    config_name = f"cofig_{id_traj}.dat"
    with open(os.path.join(out_dir,config_name),"w") as f:
        f.write(f"data_file = {data_file} \n")
        f.write(f"traj idx = {id_traj} \n")
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

    t_traj_ini = mtime.time()
    print(f"start with traj {id_traj}")
    traj = data
    ln_Ls = np.ones(C)*ln_L0

    log_file_name = f"log_trial_chains-Traj_{id_traj}.dat"
    chains_file = f"Trial_chains-Traj_{id_traj}.dat"
    chains_trial,ln_Ls_trial = execute(init_params,mean_kern,cov_kern,ln_Ls,R_trial,M_trial,C,n_estim,prior_pars,obs_li_param,known_param,log_file_name,chains_file,out_dir,traj,model_step,h,sqh)
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
    #sigma_opt = 2.38**2/n_estim*np.cov(par_complte,rowvar=False)
    #sigma_opt = np.sqrt(sigma_opt)
    #print(sigma_opt,par_end)
    
    sigma_opt = cov_kern 
    ln_Ls = ln_Ls_trial
    dfc = pd.DataFrame([])
    log_file_name = f"log_chains-Traj_{id_traj}.dat"
    chains_file = f"Chains-Traj_{id_traj}.dat"
    chains,ln_Ls= execute(par_end,mean_kern,sigma_opt,ln_Ls,R,M,C,n_estim,prior_pars,obs_li_param,known_param,log_file_name,chains_file,out_dir,traj,model_step,h,sqh)
    print("\n Done computing chains \n")

    t_traj_fin = mtime.time()
    extime = t_traj_fin-t_traj_ini
    print(f"Done with traj {id_traj} execution time {extime//60}:{extime%60} min \n \n \n")
    print("Done")
    #%%
