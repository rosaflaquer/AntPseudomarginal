#%%
import os
import pandas as pd
import numpy as np
#%%
dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
data_dir = os.path.join(proj_path,"Data","Ant_data")
data_file = "2022_Transformed_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(data_dir,data_file))

newdata_file = "2022_Tranformed_nomin-width_50-frames_40.dat"
newdatadf = pd.read_csv(os.path.join(data_dir,newdata_file))
#%%

id_olds = datadf["id_traj"].unique()
id_news = newdatadf["id_traj"].unique()

#%%
print(len(id_olds))
print(len(id_news))
print(len(id_olds)/len(id_news))
#%%
not_done = []
done = []
for iid in id_news:
    if not(iid in id_olds): not_done.append(iid)
    else: done.append(iid)
#%%
print(np.all(id_olds == done))
print(len(not_done))
#%%
name = "shorter_id_trajs.par"
with open(name,"w") as f:
    for iid in not_done:
        f.write(iid + "\n")