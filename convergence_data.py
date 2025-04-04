#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lib_MCMC import *
import lib_model

dirname = os.getcwd()
proj_path = os.path.split(dirname)[0] 
plt.style.use(os.path.join(dirname,'Estils','plots.mplstyle')) #styles file
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#%%
traj_dir = os.path.join(proj_path,"Data","Ant_data")
#data_file = "2022_Transformed_nothetarnage_width_50-frames_40.dat"
data_file = "2022_Transformed_nomin_width_50-frames_40.dat"
datadf = pd.read_csv(os.path.join(traj_dir,data_file))
main_dir = os.path.join(proj_path,"Data","Fits","LongNoPause")
folders = os.listdir(main_dir)
converged = []
id_traj = datadf["id_traj"].unique()
#%%
for name in folders:
    data_dir = os.path.join(main_dir,name)
    for tr in id_traj:

        id_folder = tr
        #print(tr,name,id_folder,is_segment+1)
        if id_folder != name[5:] : 
            #print(id_folder,"##",name,"##")
            continue
        else : id_tr = tr
        
        f = open(os.path.join(data_dir,f"pars_{id_tr}.dat"),"w")
        day_traj = datadf[datadf["id_traj"]==id_tr]
        print(len(day_traj))
        data = np.zeros((4,len(day_traj)))
        data[0] = day_traj["x"].values     
        data[1] = day_traj["y"].values     
        data[2] = day_traj["theta"].values 
        data[3] = day_traj["Time"].values


        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
        ax.plot(data[0],data[1])
        ax.set(xlabel=r"$x$",ylabel=r"$y$")
        filename = f"traj_{id_tr}.png"
        #fig.savefig(os.path.join(data_dir,filename),format="png",
        #            facecolor="w",edgecolor="w",bbox_inches="tight")


        file_name = "Trial_chains-Traj_"
        file_name = "Chains-Traj_"

        df = pd.read_csv(os.path.join(data_dir,file_name+id_tr+".dat"))
        nparam = 5
        #C = int(len(df.columns)/2)
        C = int(len(df.columns)/(nparam+2))
        M = len(df)
        chains = []
        for i in range(C):
            par_ch = []
            for j in range(nparam):
                par_ch.append(df[f"par{j}_{i}"].values)
            chains.append(par_ch)
            #chains.append([df[f"beta_{i}"].values,df[f"delta_{i}"].values])
        print(len(chains[0]))
        hatR,Se = convergence(chains,nparam,C,M)
        print(hatR,Se)

        if np.all(hatR) < 1.2: converged.append(id_tr)

        dict_params = {
            0 : [r"$\beta$",f"beta-{id_tr}.png",np.zeros(M*C)],
            1 : [r"$\delta$",f"delta-{id_tr}.png",np.zeros(M*C)],
            2 : [r"$\sigma$",f"sigma-{id_tr}.png",np.zeros(M*C)],
            3 : [r"$l$",f"l-{id_tr}.png",np.zeros(M*C)],
            4 : [r"$\phi$",f"phi-{id_tr}.png",np.zeros(M*C)],
        }

        for j in range(nparam):
            fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(11,6))
            ax.set(ylabel=dict_params[j][0],xlabel="MC step")
            for i in range(C):
                ax.plot(chains[i][j])
                dict_params[j][2] = chains[i][j]
            plt.close()
            fig.savefig(os.path.join(data_dir,dict_params[j][1]),format="png",
                        facecolor="w",edgecolor="w",bbox_inches="tight")



        def plot(xx,xlabel,nbins,filename):
            fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
            height, _, _ = ax.hist(xx,bins=nbins,density=True)
            m = np.mean(xx)
            sd = np.std(xx)
            ci = [m-sd,m+sd]
            ylims = [0, height.max()*(1+0.1)]
            ax.vlines(m,ylims[0],ylims[1],color="black")
            ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35) 
            ax.set(xlabel=xlabel,ylim=ylims)
            plt.close()
            fig.savefig(os.path.join(data_dir,filename),format="png",
                    facecolor="w",edgecolor="w",bbox_inches="tight")

        nbins = 30
        for j in range(nparam):
            plot(dict_params[j][2],dict_params[j][0],nbins,"Distr_" + dict_params[j][1])

        for i in range(nparam):
            xx = dict_params[i][2]
            mb = np.mean(xx)
            sdb = np.std(xx)
            f.write(f"{dict_params[i][0]},{mb},{sdb}\n")
            for j in range(i+1,nparam):
                yy = dict_params[j][2]
                fig = plt.figure(figsize=(15,12))
                ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
                ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
                ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)

                ax.set(xlabel=dict_params[i][0],ylabel=dict_params[j][0])
                ax.hist2d(xx,yy,bins=20,cmap="binary")
                #ax.scatter(xx,yy,s=5,alpha=0.2)

                bins = 50        
                ax.axvline(np.mean(xx),ls="--")
                ax.fill_betweenx([yy.min(),yy.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 

                height, _,_ = ax_histx.hist(xx, bins=bins, density=True,color="black")
                ax_histx.axvline(np.mean(xx),ls="--")
                ax_histx.fill_betweenx([0,height.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 

                ax.axhline(np.mean(yy),ls="--")
                md = np.mean(yy)
                sdd = np.std(yy)
                ax.fill_between([xx.min(),xx.max()], md - sdd, md + sdd, color=colors[0], alpha=0.15) 

                height, _,_ = ax_histy.hist(yy, bins=bins, density=True, orientation='horizontal',color="black")
                ax_histy.axhline(np.mean(yy),ls="--")
                ax_histy.fill_between([0,height.max()],  md - sdd, md + sdd, color=colors[0], alpha=0.15) 

                plt.close()
                filename = f"Distr-traj_{id_tr}_{i}_{j}.png"
                fig.savefig(os.path.join(data_dir,filename),format="png",
                                facecolor="w",edgecolor="w",bbox_inches="tight")
        f.close()
            



    #def sqdispl(datax,datay):
    #    if len(np.shape(datax)) > 1:
    #        return np.sum(np.sqrt((datax[1:]-datax[:-1])**2+(datay[1:]-datay[:-1])**2),axis=1)
    #    else:
    #        return np.sum(np.sqrt((datax[1:]-datax[:-1])**2+(datay[1:]-datay[:-1])**2))
#
    #def count_zero_crossings(data):
    #    signs = np.sign(data)
    #    diff_signs = np.diff(signs)
    #    if len(np.shape(data)) > 1:
    #        return np.count_nonzero(diff_signs,axis=1)
    #    else: 
    #        return np.count_nonzero(diff_signs)
#
#
    #h = 0.01 
    #v,l,phi,Mu,Sigma,th0 = day_traj["$|v|$"].mean(),13,0.9,0.0,6.5,1.0
    #param = np.array([v,Mu,th0 ])
    #icon = np.array([data[0,0],data[1,0],data[2,0]])
    #t_fin = data[3,-1]
    #Dt = data[3,1]-data[3,0]
    #print(t_fin,Dt)
    #iwr = int(Dt/h)
    #Nt = int(t_fin/h) + 1*iwr
    #Ntraj_ic = 5000
    #ndraws = 3
    #xidxs = [i for i in range(0,Ntraj_ic,3)]
    #yidxs = [i for i in range(1,Ntraj_ic,3)]
#
    #fig,axs=plt.subplots(ncols=3,nrows=3,figsize=(11*3.5,6*(ndraws+0.5)))
    #for i in range(ndraws):
    #    ks = np.array([1, 1, Sigma, l, phi])
    #    for k in range(nparam):
    #        ks[k] = rv_from_data(dict_params[k][2])
    #    data_pred = lib_model.multiple_traj(icon,h,np.sqrt(h),Nt,iwr,param,ks,Ntraj_ic)
    #    
    #    ax = axs[i][0]
    #    for j in range(0,len(data_pred),3):
    #        ax.plot(data_pred[j],data_pred[(j+1)],alpha=0.25)
#
    #    ss = np.std(data_pred[xidxs],axis=0)
    #    xm = np.mean(data_pred[xidxs],axis=0)
    #    ym = np.mean(data_pred[yidxs],axis=0)
    #    ax.plot(xm,ym,color="white")
    #    ax.fill_betweenx(ym,xm-ss,xm+ss,color="black",alpha=0.9)
    #    ax.plot(data[0,:],data[1,:],color="black")
    #    ax.set(xlabel=r"$x$",ylabel=r"$y$")
#
    #    sqd_pred = sqdispl(data_pred[xidxs],data_pred[yidxs])
    #    sqd = sqdispl(data[0],data[1])
#
    #    cros_pred = count_zero_crossings(data_pred[xidxs])
    #    cros =  count_zero_crossings(data[0])
#
    #    ax = axs[i][1]
    #    height, _,_ = ax.hist(sqd_pred,bins=50,density=True)
    #    m = np.mean(sqd_pred)
    #    sd = np.std(sqd_pred)
    #    ci = [m-sd,m+sd]
    #    ylims = [0, height.max()*(1+0.1)]
    #    ax.vlines(m,ylims[0],ylims[1],color="black")
    #    ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35) 
    #    ax.vlines(sqd,ylims[0],ylims[1],color="black",ls="--")
    #    ax.set(yscale="log",xlabel="SD")
#
    #    ax = axs[i][2]
    #    height, _,_ = ax.hist(cros_pred,bins=50,density=True)
    #    m = np.mean(cros_pred)
    #    sd = np.std(cros_pred)
    #    ci = [m-sd,m+sd]
    #    ylims = [0, height.max()*(1+0.1)]
    #    ax.vlines(m,ylims[0],ylims[1],color="black")
    #    ax.fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35)        
    #    ax.vlines(cros,ylims[0],ylims[1],color="black",ls="--")
    #    ax.set(yscale="log",xlabel="Crossings")
#
    #plt.close()
#
    #filename = f"Posterior_check-traj_{name}.png"
    #fig.savefig(os.path.join(data_dir,filename),format="png",
    #            facecolor="w",edgecolor="w",bbox_inches="tight")
#%%



#%%
main_dir = os.path.join(proj_path,"Data","Fits","Larger_sigma")
elements = os.listdir(main_dir)
folders = [element for element in elements if os.path.isdir(os.path.join(main_dir, element))]
converged = []
for name in folders:
    
    data_dir = os.path.join(main_dir,name)
    #traj_dir = os.path.join(proj_path,"Data","Ant_data")
    #data_file = "2022_Transformed_width_50-frames_40.dat"
    datadf = pd.read_csv(os.path.join(traj_dir,data_file))
    id_traj = datadf["id_traj"].unique()
    Ntrajs = len(id_traj)
    id = name[5:]
    day_traj = datadf[datadf["id_traj"]==id]
    print(len(day_traj))
    data = np.zeros((4,len(day_traj)))
    data[0] = day_traj["x"].values     
    data[1] = day_traj["y"].values     
    data[2] = day_traj["theta"].values 
    data[3] = day_traj["Time"].values

    file_name = "Chains-"

    df = pd.read_csv(os.path.join(data_dir,file_name+name+".dat"))
    nparam = 5
    #C = int(len(df.columns)/2)
    C = int(len(df.columns)/(nparam+2))
    M = len(df)
    chains = []
    for i in range(C):
        par_ch = []
        for j in range(nparam):
            par_ch.append(df[f"par{j}_{i}"].values)
        chains.append(par_ch)
    print(len(chains[0]))
    hatR,Se = convergence(chains,nparam,C,M)
    print(hatR,Se)

    if np.all(hatR < 1.1): converged.append(name)

#%%

with open(os.path.join(main_dir,"Converged.dat"), 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in converged))

#%%
fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(11*2,11))
axs[0].set(xlabel=r"$x$",ylabel=r"$y$",title="Converged")
axs[1].set(xlabel=r"$x$",ylabel=r"$y$",title="Not converged")
for i in range(2):
    axs[i].axhline(0,color="black")
    axs[i].axvline(0,color="black")
for name in folders:
    id = name[5:]
    day_traj = datadf[datadf["id_traj"]==id]
    print(id,day_traj)
    if name in converged: ii = 0
    else: ii = 1
    axs[ii].plot(day_traj["x"],day_traj["y"])

filename = f"Conv_noconv.png"
fig.savefig(os.path.join(main_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(11*2,11))
axs[0].set(xlabel=r"$t$",ylabel=r"$v_y$",title="Converged")
axs[1].set(xlabel=r"$t$",ylabel=r"$v_y$",title="Not converged")
for i in range(2):
    axs[i].axhline(0,color="black")
    axs[i].axvline(0,color="black")
for name in folders:
    id = name[5:]
    day_traj = datadf[datadf["id_traj"]==id]
    if name in converged: ii = 0
    else: ii = 1
    axs[ii].plot(day_traj["Time"],day_traj["vy"])

#%%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.set(xlabel=r"$v_{min}$",ylabel=r"$v_{max}$")
for name in folders:
    id = name[5:]
    day_traj = datadf[datadf["id_traj"]==id]
    if name in converged: color = colors[0]
    else: color = colors[1]
    ax.scatter(day_traj["$|v|$"].min(),day_traj["$|v|$"].max(),color=color)
xx = np.linspace(0,7.5,1000)
ax.plot(xx,xx,color="black")

filename = f"Vmin_vmax.png"
fig.savefig(os.path.join(main_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

#%%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(11,6))
ax.set(xlabel="Traj id",ylabel=r"$v_{max}-v_{min}$")
for i,name in enumerate(folders):
    id = name[5:]
    day_traj = datadf[datadf["id_traj"]==id]
    if name in converged: color = colors[0]
    else: color = colors[1]
    ax.scatter(i,day_traj["$|v|$"].max() - day_traj["$|v|$"].min(),color=color)
#%%
ncols = 3
nbins = 15
nrows = len(converged)//ncols+len(converged)%ncols + 1 
fig,axs = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
plt.suptitle(r"$\beta$")
old_dir = os.path.join(proj_path,"Data","Fits","Fitsnomin","mcmc","Data","Fits")
means = []
for k,name in enumerate(converged):
    data_dir = os.path.join(main_dir,name)
    data_dir = os.path.join(old_dir,name)
    id_traj = datadf["id_traj"].unique()
    file_name = "Chains-"
    df = pd.read_csv(os.path.join(data_dir,file_name+name+".dat"))
    C = int(len(df.columns)/(nparam+2))
    M = len(df)
    chains = []
    for i in range(C):
        par_ch = []
        for j in range(nparam):
            par_ch.append(df[f"par{j}_{i}"].values)
        chains.append(par_ch)
        #chains.append([df[f"beta_{i}"].values,df[f"delta_{i}"].values])
    print(len(chains[0]))
    dict_params = {
        0 : [r"$\beta$",np.zeros(M*C)],
        1 : [r"$\delta$",np.zeros(M*C)],
        2 : [r"$\sigma$",np.zeros(M*C)],
        3 : [r"$l$",np.zeros(M*C)],
        4 : [r"$\phi$",np.zeros(M*C)],
    }

    for j in range(nparam):
        for i in range(C):
            dict_params[j][1] = chains[i][j]
    xx = dict_params[0][1]
    height, _, _ = axs[k//ncols][k%ncols].hist(xx,bins=nbins,density=True,color=colors[2])
    m = np.mean(xx)
    sd = np.std(xx)
    ci = [m-sd,m+sd]
    ylims = [0, height.max()*(1+0.1)]
    axs[k//ncols][k%ncols].vlines(m,ylims[0],ylims[1],color="black")
    axs[k//ncols][k%ncols].fill_betweenx(ylims, ci[0], ci[1], color='black', alpha=0.35) 
    axs[k//ncols][k%ncols].text(0.6,0.8,name,transform=axs[k//ncols][k%ncols].transAxes)
    means.append(m)
axs[-1][-1].hist(means,bins=10,density=True,color=colors[3])
axs[-1][-1].text(0.6,0.8,"Total",transform=axs[-1][-1].transAxes)
filename = f"bs_old.png"
fig.savefig(os.path.join(main_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")


# %%
ncols = 3
nrows = len(converged)//ncols+len(converged)%ncols + 1 
fig,axs = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
for k,name in enumerate(converged):
    ax = axs[k//ncols][k%ncols]
    ax.set(xlabel=r"$x$", ylabel=r"$y$")
    ax.axvline(0,color="black")
    ax.axhline(0,color="black")
    id = name[5:]
    day_traj = datadf[datadf["id_traj"]==id]    
    xx = day_traj["x"]
    yy = day_traj["y"]
    #ax.plot(xx,yy,color=colors[2],label="Old",lw=7)
    ax.plot(xx[:-20],yy[:-20],color=colors[0]) #,label="Cut",lw=7,ls="--")
    ax.text(0.6,0.8,name,transform=ax.transAxes)
    #ax.legend(loc="best")
# %%
filename = "Trajs.png"
fig.savefig(os.path.join(main_dir,filename),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
# %%
ncols = 3
nrows = len(converged)//ncols+len(converged)%ncols + 1 
fig_1,axs_1 = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
fig_2,axs_2 = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
fig_3,axs_3 = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
fig_old_1,axs_old_1 = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
fig_old_2,axs_old_2 = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
fig_old_3,axs_old_3 = plt.subplots(ncols=ncols,nrows=nrows,figsize=(11*ncols,6*nrows))
axs = [axs_1,axs_2,axs_3]
axs_old = [axs_old_1,axs_old_2,axs_old_3]
for k,name in enumerate(converged):
    
    rw = k//ncols
    cl = k%ncols

    file_name = "Chains-"
    df = pd.read_csv(os.path.join(main_dir,name,file_name+name+".dat"))
    df_old = pd.read_csv(os.path.join(old_dir,name,file_name+name+".dat"))
    nparam = 5
    C = int(len(df.columns)/(nparam+2))
    M = len(df)
    C_old = int(len(df.columns)/(nparam+2))
    M_old = len(df)
    chains = []
    chains_old = []
    for i in range(C):
        par_ch = []
        for j in range(3):
            par_ch.append(df[f"par{j}_{i}"].values)
        chains.append(par_ch)
    for i in range(C_old):
        par_ch = []
        for j in range(3):
            par_ch.append(df_old[f"par{j}_{i}"].values)
        chains_old.append(par_ch)

    dict_params = {
            0 : [r"$\beta$", np.zeros(M*C),np.zeros(M_old*C_old)],
            1 : [r"$\delta$",np.zeros(M*C),np.zeros(M_old*C_old)],
            2 : [r"$\sigma$",np.zeros(M*C),np.zeros(M_old*C_old)],
        }

    for j in range(3):
        for i in range(C):
            dict_params[j][1] = chains[i][j]
        for i in range(C_old):
            dict_params[j][2] = chains_old[i][j]
    
    count = 0
    for i in range(3):
        xx = dict_params[i][1]
        xx_old = dict_params[i][2]
        mb = np.mean(xx)
        sdb = np.std(xx)
        mb_old = np.mean(xx_old)
        sdb_old = np.std(xx_old)
        for j in range(i+1,3):
            ax = axs[count][rw][cl]
            ax_old = axs_old[count][rw][cl]
            yy = dict_params[j][1]
            yy_old = dict_params[j][2]

            ax.set(xlabel=dict_params[i][0],ylabel=dict_params[j][0])
            ax.hist2d(xx,yy,bins=20,cmap="binary")
            ax_old.set(xlabel=dict_params[i][0],ylabel=dict_params[j][0])
            ax_old.hist2d(xx_old,yy_old,bins=20,cmap="binary")


            bins = 50        
            ax.axvline(np.mean(xx),ls="--")
            ax.fill_betweenx([yy.min(),yy.max()], mb - sdb, mb + sdb, color=colors[0], alpha=0.15) 
            ax_old.axvline(np.mean(xx_old),ls="--", color=colors[2]) 
            ax_old.fill_betweenx([yy_old.min(),yy_old.max()], mb_old - sdb_old, mb_old + sdb_old, color=colors[2], alpha=0.15) 


            ax.axhline(np.mean(yy),ls="--")
            md = np.mean(yy)
            sdd = np.std(yy)
            ax.fill_between([xx.min(),xx.max()], md - sdd, md + sdd, color=colors[0], alpha=0.15)
            ax_old.axhline(np.mean(yy_old),ls="--", color=colors[2])
            md = np.mean(yy_old)
            sdd = np.std(yy_old)
            ax_old.fill_between([xx.min(),xx.max()], md - sdd, md + sdd, color=colors[2], alpha=0.15)  

            ax.text(0.6,0.8,name,transform=ax.transAxes)
            ax_old.text(0.6,0.8,name,transform=ax_old.transAxes)
            count += 1


# %%
filename = "Marginal_b_d"
fig_1.savefig(os.path.join(main_dir,filename+".png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig_old_1.savefig(os.path.join(main_dir,filename+"_old.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

filename = "Marginal_b_s"
fig_2.savefig(os.path.join(main_dir,filename+".png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig_old_2.savefig(os.path.join(main_dir,filename+"_old.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")

filename = "Marginal_d_s"
fig_3.savefig(os.path.join(main_dir,filename+".png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
fig_old_3.savefig(os.path.join(main_dir,filename+"_old.png"),format="png",
            facecolor="w",edgecolor="w",bbox_inches="tight")
# %%
