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

dirname = os.path.dirname(os.path.abspath(__file__))
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

filtered_ids = datadf.groupby("id_traj")["Time"].max()
filtered_ids = filtered_ids[filtered_ids > 100].index
nindata = len(filtered_ids)

#result_df = datadf[(datadf["id_traj"].isin(filtered_ids)) & (datadf["Time"] == 0)][["id_traj", "x", "y","theta"]]
result_df = datadf[(datadf["id_traj"].isin(filtered_ids))][["id_traj", "Time", "x", "y", "theta", "$|v|$"]]

betas =[0.5,0.75,1.5,2,5,7.5,15]
gammas = [0.01,0.025,0.05,0.1,0.25,0.5,1,5,10] #
sigma = 75
deltas = [0.1,0.15,0.175,0.2,0.25] #[0.0,0.1,0.15,0.175,0.2,0.25]
factor = 0.5
#%%

for delta in deltas:
    print("delta",delta,"################################################################################")
    in_folders = ["Data","Comparisons","Convergence",f"v_{factor}","Delay",f"delta_{delta}"]
    in_path = os.path.join(proj_path,*in_folders)
    dict_tries = {}
    dict_converged = {}
    known_ntrajs = False
    for m,beta in enumerate(betas):
        dict_tries[beta] = [[],[]]
        dict_converged[beta] = [[],[]]
        for n,gamma in enumerate(gammas):
            print(f"delta = {delta}, beta = {beta}, gamma = {gamma} ####################")
            name =f"Beta_{beta}_gamma_{gamma}_sigma_{sigma}_delta{delta}"
            if os.path.exists(os.path.join(in_path)):
                with open(os.path.join(in_path,name + "_dict_tries.json"), 'r') as json_file:
                    dict_data_tries = json.load(json_file)
                    dict_tries[beta][0].append(gamma)
                    dict_tries[beta][1].append(dict_data_tries["Total tries"])
                    dict_converged[beta][0].append(gamma)
                    dict_converged[beta][1].append(dict_data_tries["Total converged"])
                    if not known_ntrajs:
                        Ntraj = dict_data_tries["Total converged"] + dict_data_tries["Total deleted"]
                        known_ntrajs = True 
            else:
                print("The path does not exist")

    no_delayed_tries = []
    no_delayed_converged = []
    known_ntrajs = False
    in_folders = ["Data","Comparisons","Convergence",f"v_{factor}","Regular",f"delta_{delta}"]
    in_path = os.path.join(proj_path,*in_folders)
    for m,beta in enumerate(betas):
        print(f"beta = {beta} ########################################")
        name =f"Beta_{beta}_sigma_{sigma}_delta{delta}"
        if os.path.exists(os.path.join(in_path)):
            with open(os.path.join(in_path,name + "_dict_tries.json"), 'r') as json_file:
                dict_data_tries = json.load(json_file)
                no_delayed_tries.append(dict_data_tries["Total tries"])
                no_delayed_converged.append(dict_data_tries["Total converged"])
                if not known_ntrajs:
                    Ntraj_conv = dict_data_tries["Total converged"] + dict_data_tries["Total deleted"]
                    known_ntrajs = True 
        else:
            print("The path does not exist")

    no_delayed_novel_tries = []
    no_delayed_novel_converged = []
    known_ntrajs = False
    in_folders = ["Data","Comparisons","Convergence",f"v_{factor}","Regular_novel",f"delta_{delta}"]
    in_path = os.path.join(proj_path,*in_folders)
    for m,beta in enumerate(betas):
        print(f"beta = {beta} ########################################")
        name =f"Beta_{beta}_sigma_{sigma}_delta{delta}"
        if os.path.exists(os.path.join(in_path)):
            with open(os.path.join(in_path,name + "_dict_tries.json"), 'r') as json_file:
                dict_data_tries = json.load(json_file)
                no_delayed_novel_tries.append(dict_data_tries["Total tries"])
                no_delayed_novel_converged.append(dict_data_tries["Total converged"])
                if not known_ntrajs:
                    Ntraj_conv = dict_data_tries["Total converged"] + dict_data_tries["Total deleted"]
                    known_ntrajs = True 
        else:
            print("The path does not exist")



    #fig, ax = plt.subplots(nrows=len(betas),ncols=1,figsize=(6, 3*len(betas)),sharex=True)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8, 6))
    for i,beta in enumerate(betas):
        ax.plot(dict_tries[beta][0],dict_tries[beta][1],label=r"$\beta=${}".format(beta),
                marker='o',linestyle='--',color=colors[i])
        ax.axhline(no_delayed_tries[i],color=colors[i])
        ax.axhline(no_delayed_novel_tries[i],color=colors[i],linestyle=":")
    ax.set(
        ylabel = r"\# tries",
        yscale = "log",
        xlabel = r"$\gamma$",
        xscale = "log",
        title = r"$\delta$ = {}".format(delta),
        )
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    out_path = os.path.join(proj_path,"Data","Comparisons","Convergence",f"v_{factor}")
    out_name = f"Tries_vs_gamma_delta_{delta}_sigma_{sigma}_v_{factor}_novel"
    fig.savefig(os.path.join(out_path,out_name + ".jpg"),bbox_inches='tight')


    #fig, ax = plt.subplots(nrows=len(betas),ncols=1,figsize=(6, 3*len(betas)),sharex=True)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8, 6))
    for i,beta in enumerate(betas):
        ax.plot(dict_converged[beta][0],dict_converged[beta][1],label=r"$\beta=${}".format(beta),
                marker='o',linestyle='--',color=colors[i])
        ax.axhline(no_delayed_converged[i],color=colors[i])
        ax.axhline(no_delayed_novel_converged[i],color=colors[i],linestyle=":")
    ax.axhline(Ntraj,color="black")
    ax.set(
        ylabel = r"\# converged trajs",
        xlabel = r"$\gamma$",
        xscale = "log",
        title = r"$\delta$ = {}".format(delta),
        )
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    out_path = os.path.join(proj_path,"Data","Comparisons","Convergence",f"v_{factor}")
    out_name = f"Conv_vs_gamma_delta_{delta}_sigma_{sigma}_v_{factor}_novel"
    fig.savefig(os.path.join(out_path,out_name + ".jpg"),bbox_inches='tight')
    plt.show()
# %%


#plot histograms

for delta in deltas:
    fig,ax   = plt.subplots(nrows=len(betas),ncols=(len(gammas)+1),figsize=(7*(len(gammas)+1),8.5*len(betas)))
    fig2,ax2 = plt.subplots(nrows=len(betas),ncols=(len(gammas)+1),figsize=(7*(len(gammas)+1),8.5*len(betas)))
    print(delta)
    in_folders = ["Data","Comparisons","Convergence",f"v_{factor}","Delay",f"delta_{delta}"]
    in_path = os.path.join(proj_path,*in_folders)
    for m,beta in enumerate(betas):
        print(beta)
        for n,gamma in enumerate(gammas):
            print(f"beta = {beta}, gamma = {gamma} ########################################")
            name =f"Beta_{beta}_gamma_{gamma}_sigma_{sigma}_delta{delta}"
            if os.path.exists(os.path.join(in_path)):
                data_x = np.load(os.path.join(in_path,name + "_v_x.npz"))
                data_y = np.load(os.path.join(in_path,name + "_v_y.npz"))
            else:
                print("The path does not exist")
                continue
            try:
                ax[m][n].plot((data_x["bins1_x"][1:]+data_x["bins1_x"][:-1])*0.5,data_x["n1_x"],color="black")
                ax[m][n].plot((data_x["bins2_x"][1:]+data_x["bins2_x"][:-1])*0.5,data_x["n2_x"],lw=5,)
                kullback = kl(data_x["n1_x"],data_x["n2_x"],data_x["bins1_x"],data_x["bins2_x"])
                ax[m][n].text(0.15,0.8,f"kl = {kullback:.3f}",transform=ax[m][n].transAxes)
                ax[m][n].text(0.65,0.8,r"$\beta$ = {}".format(beta),transform=ax[m][n].transAxes)
                ax[m][n].text(0.65,0.6,r"$\gamma$ = {}".format(gamma),transform=ax[m][n].transAxes)
                ax[m][n].text(0.65,0.4,r"$\delta$ = {}".format(delta),transform=ax[m][n].transAxes)
                ax[m][n].text(0.65,0.2,r"$\sigma$ = {}".format(sigma),transform=ax[m][n].transAxes)

                ax2[m][n].plot((data_y["bins1_x"][1:]+data_y["bins1_x"][:-1])*0.5,data_y["n1_x"],color="black")
                ax2[m][n].plot((data_y["bins2_x"][1:]+data_y["bins2_x"][:-1])*0.5,data_y["n2_x"],lw=5,)
                kullback = kl(data_x["n1_x"],data_x["n2_x"],data_x["bins1_x"],data_x["bins2_x"])
                ax2[m][n].text(0.15,0.8,f"kl = {kullback:.3f}",transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.65,0.8,r"$\beta$ = {}".format(beta),transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.65,0.6,r"$\gamma$ = {}".format(gamma),transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.65,0.4,r"$\delta$ = {}".format(delta),transform=ax2[m][n].transAxes)
                ax2[m][n].text(0.65,0.2,r"$\sigma$ = {}".format(sigma),transform=ax2[m][n].transAxes)
            
            except Exception as e:
                print(f"Error in building the histogram: {e}")
                pass

    in_folders = ["Data","Comparisons","Convergence",f"v_{factor}","Regular",f"delta_{delta}"]
    in_path = os.path.join(proj_path,*in_folders)
    n=-1
    for m,beta in enumerate(betas):
        print(f"beta = {beta} ########################################")
        name =f"Beta_{beta}_sigma_{sigma}_delta{delta}"
        data_x = np.load(os.path.join(in_path,name + "_v_x.npz"))
        data_y = np.load(os.path.join(in_path,name + "_v_y.npz"))
        ax[m][n].plot((data_x["bins1_x"][1:]+data_x["bins1_x"][:-1])*0.5,data_x["n1_x"],color="black")
        ax[m][n].plot((data_x["bins2_x"][1:]+data_x["bins2_x"][:-1])*0.5,data_x["n2_x"],lw=5,color=colors[2])
        kullback = kl(data_x["n1_x"],data_x["n2_x"],data_x["bins1_x"],data_x["bins2_x"])
        ax[m][n].text(0.15,0.8,f"kl = {kullback:.3f}",transform=ax[m][n].transAxes)
        ax[m][n].text(0.65,0.8,r"$\beta$ = {}".format(beta),transform=ax[m][n].transAxes)
        ax[m][n].text(0.65,0.6,r"$\gamma = \infty$",transform=ax[m][n].transAxes)
        ax[m][n].text(0.65,0.4,r"$\delta$ = {}".format(delta),transform=ax[m][n].transAxes)
        ax[m][n].text(0.65,0.2,r"$\sigma$ = {}".format(sigma),transform=ax[m][n].transAxes)


        ax2[m][n].plot((data_y["bins1_x"][1:]+data_y["bins1_x"][:-1])*0.5,data_y["n1_x"],color="black")
        ax2[m][n].plot((data_y["bins2_x"][1:]+data_y["bins2_x"][:-1])*0.5,data_y["n2_x"],lw=5,color=colors[2])
        kullback = kl(data_x["n1_x"],data_x["n2_x"],data_x["bins1_x"],data_x["bins2_x"])
        ax2[m][n].text(0.15,0.8,f"kl = {kullback:.3f}",transform=ax2[m][n].transAxes)
        ax2[m][n].text(0.65,0.8,r"$\beta$ = {}".format(beta),transform=ax2[m][n].transAxes)
        ax2[m][n].text(0.65,0.6,r"$\gamma = \infty$",transform=ax2[m][n].transAxes)
        ax2[m][n].text(0.65,0.4,r"$\delta$ = {}".format(delta),transform=ax2[m][n].transAxes)
        ax2[m][n].text(0.65,0.2,r"$\sigma$ = {}".format(sigma),transform=ax2[m][n].transAxes)

    out_path = os.path.join(proj_path,"Data","Comparisons","Convergence",f"v_{factor}")
    out_name = f"Hist_delta_{delta}_sigma_{sigma}_v_{factor}"
    fig.savefig(os.path.join(out_path,"x"+out_name + ".jpg"),bbox_inches='tight')
    fig2.savefig(os.path.join(out_path,"y"+out_name + ".jpg"),bbox_inches='tight')

# %%


#Prova

beta = 0.5
gamma = 0.025
sigma = 50
delta = 0.15
name =f"Beta_{beta}_sigma_{sigma}_delta{delta}"
in_folders = ["Data","Comparisons","Convergence",f"v_{factor}","Delay",f"delta_{delta}"]
in_path = os.path.join(proj_path,*in_folders)
if os.path.exists(os.path.join(in_path)):
    with open(os.path.join(in_path,name + "_dict_tries.json"), 'r') as json_file:
        dict_data_tries = json.load(json_file)
    print("No delay Tries",dict_data_tries["Total tries"])
    print("No delay converged",dict_data_tries["Total converged"])

#%%
name =f"Beta_{beta}_gamma_{gamma}_sigma_{sigma}_delta{delta}"
in_folders = ["Data","Comparisons","Convergence","Delay",f"delta_{delta}"]
in_path = os.path.join(proj_path,*in_folders)
if os.path.exists(os.path.join(in_path)):
    with open(os.path.join(in_path,name + "_dict_tries.json"), 'r') as json_file:
        dict_data_tries = json.load(json_file)
    print("Delayed Tries",dict_data_tries["Total tries"])
    print("Delayed converged",dict_data_tries["Total converged"])
    print("Delayed converged",dict_data_tries["Which deleted"])
    print("Delayed deleted",dict_data_tries["Total deleted"])
# %%
