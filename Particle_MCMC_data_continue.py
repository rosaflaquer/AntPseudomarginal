#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lib_MCMC import *
from lib_model import model_step
import time as mtime

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
data_dir = os.path.join(proj_path,"Data","Ant_data")

#with open(os.path.join(dirname,"continue_ids.par"),"r") as f:
#    continue_folders =  f.readlines() #list of file ids.

continue_folders = ["04_oct_222_7.0"]

file_name = "Trial_chains-Traj_"
    #file_name = "Chains-Traj_"

for item in continue_folders:
    id_traj = item.strip("\n")
    id_folder = id_traj
    is_segment = id_folder.find("0_s")
    id_folder = id_folder[:is_segment+1]
    out_dir_list = ["Data","Fits","Long_NoPause",f"Traj_{id_traj}"]
    out_dir = proj_path
    for directory in out_dir_list:
        out_dir = os.path.join(out_dir,directory)
        if not(os.path.exists(out_dir)): os.mkdir(out_dir)

    #read configuration
    config_f = open(os.path.join(out_dir,"cofig_"+item+".dat"),"r")
    data_file = config_f.readline().split()[-1]
    traj_idx = config_f.readline().split()[-1]
    seed = int(config_f.readline().split()[-1])
    numthreads = int(config_f.readline().split()[-1])
    R_trial = int(config_f.readline().split()[-1])
    M_trial = int(config_f.readline().split()[-1])
    R = int(config_f.readline().split()[-1])
    M = int(config_f.readline().split()[-1])
    ln_L0 = float(config_f.readline().split()[-1]) #no needed
    C = int(config_f.readline().split()[-1])
    n_estim = int(config_f.readline().split()[-1])
    for i in range(C): 
        config_f.readline() #no need to access the beggining parameters of the original chains
    h = float(config_f.readline().split()[-1])
    v = float(config_f.readline().split()[-1])
    l = float(config_f.readline().split()[-1])
    phi = float(config_f.readline().split()[-1])
    Mu = float(config_f.readline().split()[-1])
    Sigma = float(config_f.readline().split()[-1])
    th0 = float(config_f.readline().split()[-1])
    array_str = config_f.readline().split('=')[1].strip()
    obs_li_param = np.fromstring(array_str.strip('[]'), sep=' ')
    cov_kern = np.zeros((n_estim,n_estim))
    for i in range(n_estim):
        array_str = config_f.readline().split('=')[1].strip()
        cov_kern[i] = np.fromstring(array_str.strip('[]'), sep=' ')
    prior_pars = np.zeros((n_estim,3))
    for i in range(n_estim):
        array_str = config_f.readline().split('=')[1].strip()
        prior_pars[i] = np.fromstring(array_str.strip('[]'), sep=' ')
    neq = int(config_f.readline().split()[-1])
    neq_dat = int(config_f.readline().split()[-1])
    config_f.close()
    sqh = np.sqrt(h)
    mean_kern = np.zeros(n_estim)
    known_param = np.array([v,Mu,th0])

    #load data from trajectories
    datadf = pd.read_csv(os.path.join(data_dir,data_file))
    len_trajs = len(datadf[datadf["id_traj"]==id_traj])
    data = np.zeros((4,len_trajs))
    day_traj = datadf[datadf["id_traj"]==id_traj]
    if len(day_traj) == 0: continue
    
    #load data from last execution
    df_prvious = pd.read_csv(os.path.join(out_dir,file_name+id_traj+".dat"))
    init_params = np.zeros((C,n_estim))
    ln_Ls = np.zeros(C)
    for i in range(C):
        ln_Ls[i] = df_prvious[f"ln_L_{i}"].iloc[-1]
        for j in range(n_estim):
            init_params[i][j] = df_prvious[f"par{j}_{i}"].iloc[-1]
    t_traj_ini = mtime.time()
    print(f"start with traj {id_traj}")
    traj = data

    sigma_opt = cov_kern 
    dfc = pd.DataFrame([])
    log_file_name = f"log_chains-Traj_{id_traj}_{M}.dat"
    chains_file = f"Chains-Traj_{id_traj}_{M}.dat"
    chains,ln_Ls,accepted= execute(init_params,mean_kern,sigma_opt,ln_Ls,R,M,C,n_estim,prior_pars,obs_li_param,known_param,log_file_name,chains_file,out_dir,traj,model_step,h,sqh,neq,neq_dat)
    print("\n Done computing chains \n", accepted)

    t_traj_fin = mtime.time()
    extime = t_traj_fin-t_traj_ini
    print(f"Done with traj {id_traj} execution time {extime//60}:{extime%60} min \n \n \n")
    print("Done")



# %%
