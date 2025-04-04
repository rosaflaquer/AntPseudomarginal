#%%
import numpy as np
import os
import matplotlib.pyplot as plt
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


t_fin = 200
dwr = 1 
v = 6
l = 12.8
phi = 0.95
Mu = 0.0
Sigma = 75
th0 = 1.0
delta = 0.175
beta = 1.5*0.5
gamma = 10

#Model extended
param = np.array([v,Mu,th0,Sigma,l,phi,delta]) #"known" model parameters
ks = np.array([beta,gamma]) #This is what we want to inffer!
betas = [0.5,0.75,1.5,2,5,7.5,15]
gammas = [0.01,0.025,0.05,0.1,0.25,0.5,1,5,10] #
sigma = Sigma

out_folders = ["Data","Comparisons","Convergence","Delay",f"delta_{delta}"]
out_path = os.path.join(proj_path,*out_folders)
for m,beta in enumerate(betas):
    print(beta)
    for n,gamma in enumerate(gammas):
        name = f"Beta_{beta}_gamma_{gamma}_sigma_{sigma}_delta{delta}"
        paht_datax = os.path.join(out_path,name + "_v_x.npz")
        paht_datay = os.path.join(out_path,name + "_v_y.npz")
        if os.path.exists(paht_datax and os.path.exists(paht_datay)):
            data_x = np.load(paht_datax)
            data_y = np.load(paht_datay)
        else :
            print("Data path doesn't exist")
        with open(os.path.join(out_path,name + "_dict_tries.json"), 'r') as json_file:
            dict_tries = json.load(json_file)

#%%
print(beta,gamma)
# %%
dict_tries
# %%
