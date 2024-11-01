#%%
import numpy as np
import os
import matplotlib.pyplot as plt
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
plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#%%
#1) Load observations
data_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_width_50-frames_40.dat"
#data_file = "CM_Tranformed_2022_Alldays-width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))
id_traj_list = datadf["id_traj"].unique()
idx = 0
id_traj = id_traj_list[idx]
id_traj = "27_sep_313_2.0"
out_dir = os.path.join(proj_path,"Data","Fits",f"Traj_{id_traj}")
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

ntraj_data = len(id_traj)
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
cm = ax.scatter(data[0,:],data[1,:],c=vs)    
ax.set(xlabel=r"$x$",ylabel=r"$y$",title=id_traj)
ax.axvline(0,color="black")
plt.colorbar(cm,ax=ax)
filename = f"Traj_{id_traj}.png"
fig.savefig(os.path.join(out_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
plt.show()
plt.close()
#%%

seed = 42
np.random.seed(seed)
R_trial = 250 #number of particles trial chains
M_trial = 6000 #number of MC steps trial chains
R = 250 #number of particles
M = 50000 #number of MC steps
ln_L0 = 0
C = 4 #number of chains
n_estim = 2

min_beta, max_beta = 0,1
min_delta, max_delta = 0.01, 0.25
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
cov_kern = np.array([[prior_var(prior_pars[0])/1000,0],
                     [0,prior_var(prior_pars[1])/1000]])
print(cov_kern)

obs_li_param = np.array([24,24,5*np.pi/180]) #From bbox histograms.

h = 0.1
sqh = np.sqrt(h)
l,phi,Mu,Sigma,th0 = 13,0.9,0.0,6.5,1.0
known_param = np.array([v,l,phi,Mu,Sigma,th0])

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
    f.write(f"min_beta, max_beta = {min_beta}, {max_beta}\n")
    f.write(f"min_delta, max_delta = {min_delta}, {max_delta}\n")
    f.write(f"h = {h} \n")
    f.write(f"v,l,phi,Mu,Sigma,th0 = {v},{l},{phi},{Mu},{Sigma},{th0} \n")
    f.write(f"obs_li_param = {obs_li_param} \n")

#%%

def execute(init_params,cov_kern,log_file,chains_file,dfc,R,M,chains,ln_Ls):
    for i in range(C):
        param_0 = init_params[i]
        print("\n Chain",i,"init param", param_0,"####################"*10,"\n")
        time_init = mtime.time()
        parameters_0,naccept,ln_L = ln_MCMC(M,n_estim,param_0,ln_Ls[i],proposal_kernel_rv,prior_dist,prior_pars,mean_kern,cov_kern,ln_bootstrap,traj,R,model_step,log_obs_like,obs_li_param,h,sqh,known_param)
        time_fin = mtime.time()
        extime = time_fin - time_init
        print("execution time", extime, "s", extime/60, "min")
        chains.append([parameters_0[:,0],parameters_0[:,1]])
        ln_Ls[i] = ln_L[-1]
        dfc[f"beta_{i}" ] = parameters_0[:,0]
        dfc[f"delta_{i}"] = parameters_0[:,1]
        dfc[f"ln_L_{i}"] = naccept
        dfc[f"accep_{i}"] = ln_L
        dfc.to_csv(os.path.join(out_dir,chains_file),index=False)
        log_file.write(f"chain = {i} ################################################## \n")
        log_file.write(f"execution time = {extime} \n")


# %%


t_traj_ini = mtime.time()
print(f"start with traj {id_traj}")
traj = data
chains_trial = []
ln_Ls_trial = np.ones(C)*ln_L0
dft = pd.DataFrame([])
log_file = open(os.path.join(out_dir,f"log_trial_chains-Traj_{id_traj}.dat"),"w")
chains_file = f"Trial_chains-Traj_{id_traj}.dat"
execute(init_params,cov_kern,log_file,chains_file,dft,R_trial,M_trial,chains_trial,ln_Ls_trial)
dft.to_csv(os.path.join(out_dir,chains_file),index=False)
log_file.close()
print("\n Done computing trial chains \n")

par_complte = np.zeros(((M_trial+1)*C,n_estim))
par_end = []
for i in range(C):
    par_complte[i*(M_trial+1):(M_trial+1)*(i+1),0]  = chains_trial[i][0]
    par_complte[i*(M_trial+1):(M_trial+1)*(i+1),1]  = chains_trial[i][1]
    par_end.append([chains_trial[i][0][-1],chains_trial[i][1][-1],])
#
#sigma_opt = 2.38**2/n_estim*np.cov(par_complte,rowvar=False)
#print(sigma_opt,par_end)
#
sigma_opt = cov_kern 
chains = []
ln_Ls = ln_Ls_trial
dfc = pd.DataFrame([])
log_file = open(os.path.join(out_dir,f"log_chains-Traj_{id_traj}.dat"),"w")
chains_file = f"Chains-Traj_{id_traj}.dat"
execute(par_end,sigma_opt,log_file,chains_file,dfc,R,M,chains,ln_Ls)
dfc.to_csv(os.path.join(out_dir,chains_file),index=False)
log_file.close()
hatR,ESS = convergence(chains,n_estim,C,M+1)
log_file = open(os.path.join(out_dir,f"log_chains-Traj_{id_traj}.dat"),"a")
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