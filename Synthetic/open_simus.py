#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import set_num_threads
import pandas as pd
import sys
sys.path.append('../')
import lib_model_extended as lib
import json

def kl(p,q,binsp,binsq):
    assert(np.all(binsp==binsq))
    p_2 = np.where(p>0)[0]
    q_2 = np.where(q>0)[0]
    idxs = np.intersect1d(p_2,q_2)
    p = p[idxs]
    q = q[idxs]
    return np.sum(p*np.log(p/q))


#%%

numthreads = 12
set_num_threads(numthreads)
print("Using", numthreads)

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(proj_path,'Estils','plots.mplstyle')) #styles file
proj_path = os.path.split(proj_path)[0] 
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# %%
data_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_nothetarnage_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))


#%%
#Generate a trajectory from the data #TODO: substitute this step for actual data one working.

t_fin = 200
dwr = 1 
v = 6
l = 12.8
phi = 0.95
Mu = 0.0
Sigma = 75
th0 = 1.0
delta = 0.15
beta = 1.5*0.5
gamma = 10

#Model extended
param = np.array([v,Mu,th0,Sigma,l,phi,delta]) #"known" model parameters
ks = np.array([beta,gamma]) #This is what we want to inffer!

#Simulation setup.
h = 0.1 #time step
Nt = int(t_fin/h)
iwr = int(dwr/h)
Ntraj = 100000 
filtered_ids = datadf.groupby("id_traj")["Time"].max()
filtered_ids = filtered_ids[filtered_ids > 100].index
nindata = len(filtered_ids)

#result_df = datadf[(datadf["id_traj"].isin(filtered_ids)) & (datadf["Time"] == 0)][["id_traj", "x", "y","theta"]]
result_df = datadf[(datadf["id_traj"].isin(filtered_ids))][["id_traj", "Time", "x", "y", "theta", "$|v|$"]]

betas = [0.5,0.75] #,1.5,2,5,7.5,15]
gammas = [0.01,0.025] #,0.05,0.1,0.25] #
sigma = Sigma
deltas = [0.175]
#%%

for delta in deltas:
    print(delta)
    out_folders = ["Data","Comparisons","Convergence","Delay",f"delta_{delta}"]
    out_path = proj_path
    for folder in out_folders:
        out_path = os.path.join(out_path,folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    for m,beta in enumerate(betas):
        print(beta)
        for n,gamma in enumerate(gammas):
            print(f"beta = {beta}, gamma = {gamma} ########################################")
            name =f"Beta_{beta}_gamma_{gamma}_sigma_{sigma}_delta{delta}"
            if os.path.exists(os.path.join(out_path)):
                data_x = np.load(os.path.join(out_path,name + "_v_x.npz"))
                data_y = np.load(os.path.join(out_path,name + "_v_y.npz"))
                with open(os.path.join(out_path,name + "_dict_tries.json"), 'r') as json_file:
                    dict_tries = json.load(json_file)
            else:
                print("The path does not exist")
# %%
cols = data_x.files
# %%
for idx in cols:
    print(idx)
    print(data_x[idx])
# %%
dict_tries
# %%
conv_ids = dict_tries["Which converged"].split(",")
print(conv_ids)
for id in conv_ids:
    print(id,dict_tries[id])
# %%
