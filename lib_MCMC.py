#%%
import numpy as np
from numba import njit,prange
from numba_stats import norm,uniform

@njit 
def getIC(data,n):
    """
    Returns the IC of a trajectory stored in data.
    """
    ics = np.ones((n,3))
    ics[:,0] = data[0,0]
    ics[:,1] = data[1,0]
    ics[:,2] = data[2,0]
    return ics


@njit
def rv_from_data(data,nbins=100):
    probs,bins,cum = histo_from_data(data,nbins=nbins)
    value = distr_from_histo(bins,cum)
    return value

@njit
def distr_from_histo(histo_bin,histo_cums):
    idx = np.where(histo_cums>=np.random.uniform(0,1))[0]
    print("idx",idx)
    return histo_bin[idx[0]]

@njit
def histo_from_data(data,nbins=50):
    Ndat = len(data)
    probs,bins = np.histogram(data,nbins)
    probs = probs/(Ndat*(bins[1]-bins[0]))
    bins = (bins[1:]+bins[:-1])*0.5 #From edges to centers.
    cum = np.cumsum(probs)*(bins[1]-bins[0])
    return probs,bins,cum

@njit
def prior_rvs(prior_pars):
    print(prior_pars)
    par_beta = prior_pars[0]
    par_gamma = prior_pars[1]
    values = np.zeros(2)
    values[0] = np.random.uniform( par_beta[0], par_beta[1])
    values[1] = np.random.uniform(par_gamma[0],par_gamma[1])
    return values

@njit
def prior_dist(vals,prior_pars):
    par_beta = prior_pars[0]
    par_delta = prior_pars[1]
    beta_val,delta_val = vals[0], vals[1]
    if beta_val > par_beta[0] and beta_val < par_beta[1] and delta_val > par_delta[0] and delta_val < par_delta[1]:
        pb = 1/(par_beta[1]-par_beta[0])
        pd = 1/(par_delta[1]-par_delta[0])
        value = pb*pd
    else:
        value = 0
    return value

@njit
def proposal_kernel_rv(mean_kern,cov_kern,n_estim,n=1):
    """
    rv from a multivariate normal.
    """
    perturbation = 0.0001
    cov_kern_p = cov_kern + perturbation*np.identity(n_estim)
    L = np.linalg.cholesky(cov_kern_p)
    u = np.random.normal(loc=0, scale=1, size=n_estim*n).reshape(n_estim, n)
    pp = mean_kern.reshape(n_estim, 1) + np.dot(L, u)
    return np.transpose(pp)[0]

@njit
def obs_like(X,Y,Nvar,sigma):
    """
    Observation likelyhood from data
    """
    obsl = 1.0
    sqrt2pi = np.sqrt(2*np.pi)
    for i in range(Nvar):
        point = X[i]-Y[i]
        #ol = norm.pdf(point,loc=0.0,scale=sigma[i])[0]
        ol = np.exp(-point**2/(2*sigma[i]**2))/(sqrt2pi*sigma[i])
        obsl = obsl*ol
    return obsl

@njit
def MCMC(M,n_estim,param_0,L0,proposal_kernel,prior,prior_pars,mean_kern,cov_kern,estimate_L,data,R,model,obs_li_f,obs_li_param,h,sqh,known_param):
    parameters = np.zeros((M+1,n_estim)) 
    Ls = np.zeros(M+1)
    parameters[0] = param_0
    Ls[0] = L0
    for m in range(1,M+1): #montecarlo steps
        valid = False
        while not(valid):
            proposal = parameters[m-1] + proposal_kernel(mean_kern,cov_kern,n_estim)
            val_prior_prop = prior(proposal,prior_pars)
            if val_prior_prop > 0 : valid = True
        L_star = estimate_L(L0,data,R,model,proposal,obs_li_f,obs_li_param,h,sqh,known_param) #For this implementation, pseudoespectral method with a bootstrap particle filter.
        alpha = np.min(np.array([1,(L_star*val_prior_prop)/(Ls[m-1]*prior(parameters[m-1],prior_pars))]))
        u = np.random.uniform(0,1)
        if u < alpha: 
            parameters[m] = proposal
            Ls[m] = L_star
        else:
            parameters[m] = parameters[m-1]
            Ls[m] = Ls[m-1]
        print("Likelihood",Ls[m],L_star)
    return parameters,Ls


@njit(parallel=True)
def bootstrap(Lstar,data,R,model,proposal,obs_li_f,obs_li_param,h,sqh,known_param):
    ncoord_t,N = np.shape(data) #number of coordenates and time observations
    Ncoord = ncoord_t-1 #take out the time.
    state = getIC(data[:Ncoord],R) #initialize the R particles. 
    ts = data[Ncoord] #get the data points.
    weights = np.zeros(R) #initialize the weights
    for i in range(N-1): #loop over the observaitons
        t = ts[i]
        Dt = ts[i+1] - t #step beteewn observations
        Nt = int(Dt/h) #steps to perform between the timecourse data and the h steps of the simulation
        Y = data[:Ncoord,i+1]  #data points
        for k in prange(R): #loop over the particles
        #for k in range(R): #loop over the particles
            state[k],t_ev = model(state[k],t,h,sqh,Nt,known_param,proposal) #evolve from t_i to t_i+1
            w =  obs_li_f(state[k],Y,Ncoord,obs_li_param)
            weights[k] = w
        wsum = np.sum(weights)
        if wsum == 0: 
            Lstar=0
            #print(i,state,Y)
            break
        Lstar = Lstar*wsum/R #TODO: make it log once this works.
        w_norm = weights/wsum
        cum_w = np.cumsum(w_norm)
        state_resampled = np.zeros((R,Ncoord))
        for k in prange(R):
        #for k in range(R):
            state_resampled[k] = state[np.where(cum_w>=np.random.uniform(0,1))[0][0]]
        print(i,Lstar,Y,state,state_resampled)
        state = state_resampled
    return Lstar

