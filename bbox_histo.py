#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dirname = os.path.dirname(os.path.abspath(__file__)) #script direcotry
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join( os.path.split(proj_path)[0],'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

data_path = os.path.join(proj_path,"Data","Ant_data")
files = os.listdir(data_path)
for data_file in files:
    if data_file[-3:] != "csv": continue
    df = pd.read_csv(os.path.join(data_path,data_file))
    df = df[df[" AspectRatio"]<6].copy()
    df = df[df[" AspectRatio"]>1.2].copy()
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
    height, _,_ =ax.hist(df[" AspectRatio"],bins=100,density=True)
    mb = df[" AspectRatio"].mean()
    sdb = df[" AspectRatio"].std()
    ax.fill_betweenx([0,height.max()], mb - sdb, mb + sdb, color="black", alpha=0.15) 
    ax.vlines(mb,0,height.max(),color="black",label=r"$\langle AR \rangle =$ {:.1f} $\pm$ {:.1f}".format(mb,sdb))
    ax.legend()
    ax.set(xlabel=r"$AR$")
    plt.show()
    filename = data_file[:-4] + "AR_hist.png"
    fig.savefig(os.path.join(data_path,"Plots",filename),format="png",
        facecolor="w",edgecolor="w",bbox_inches="tight")
    
    #es = df[" AspectRatio"].std()
    #print(data_file,df[" AspectRatio"].mean(),es)
    #fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
    #df["diff"] = df.groupby(["# Trajectory"])[" Time"].diff(1)
    #df = df[df["diff"] < 1.5].copy()
    #df["dx"] = df.groupby(["# Trajectory"])[" x"].diff(1)
    #df["dy"] = df.groupby(["# Trajectory"])[" y"].diff(1)
    #df["eth"] = np.sqrt((df["dx"]/(df["dx"]**2+df["dy"]**2))**2*es**2 + (df["dy"]/(df["dx"]**2+df["dy"]**2))**2*es**2  )
    #ax.hist(df["eth"],bins=100,density=True,)
    #ax.set(yscale="log")
    #ax.axvline(df["eth"].mean(),color="black")
    #ax.axvline(df["eth"].mean()+df["eth"].std(),color="grey")
    #ax.axvline(df["eth"].mean()-df["eth"].std(),color="grey")
    #print(data_file,df["eth"].mean(),df["eth"].std())
    #plt.show()

#%%

data_file2 = "2022_Alldays-width_50-frames_40.dat"
df2 = pd.read_csv(os.path.join(data_path,data_file2))

#%%
df2_f = df2[df2["diff"]<1.5]
df2_s = df2_f.shift(1)
dx = df2_s["x"] - df2_f["x"]
dy = df2_s["y"] - df2_f["y"]
e = 24
eth = np.sqrt((dx/(dx**2+dy**2))**2*e**2 + (dy/(dx**2+dy**2))**2*e**2  )

#%%


fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.hist(eth,bins=100,density=True,)
plt.show()
print(eth.std())

#%%
df = df[df["diff"] < 1.5].copy()
df["dx"] = df.groupby(["# Trajectory"])[" x"].diff(1)
df["dy"] = df.groupby(["# Trajectory"])[" y"].diff(1)
