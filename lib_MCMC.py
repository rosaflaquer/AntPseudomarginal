#%%
import numpy as np
from numba import njit,prange
from scipy.special import ndtri

@njit 
def getIC(data,n):
    """
    Returns the IC of a trajectory stored in data. #Can be from a distribution if we want.
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
    n_estim = len(prior_pars)
    values = np.zeros(n_estim)
    for i in range(n_estim):
        values[i] = np.random.uniform(prior_pars[i][0],prior_pars[i][1])
    return values

@njit
def prior_var(prior_pars):
    """
    Variance of the uniform distribution.
    """
    if prior_pars[2] == 0:
        return (1/12)*(prior_pars[1]-prior_pars[0])**2
    if prior_pars[2] == 1:
        return prior_pars[1]**2

@njit
def prior_dist(vals,prior_pars):
    n_estim = len(prior_pars)
    value = 1 
    for i in range(n_estim):
        if prior_pars[i][2] == 0: #uniform distribution
            in_range = (vals[i] > prior_pars[i][0] and vals[i] < prior_pars[i][1])
            if in_range:
                value = value*(1/(prior_pars[i][1]-prior_pars[i][0]))
            else:
                value = 0
                return value
        elif prior_pars[i][2] == 1: #gaussian distribution
            value = value*(1/np.sqrt(2*np.pi*prior_pars[i][1]**2))*np.exp(-0.5*(prior_pars[i][0]-vals[i])**2/prior_pars[i][1]**2)
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
    sqrt2pi = np.sqrt(2*np.pi)
    point = X-Y
    obsl = np.prod(np.exp(-0.5*point**2/sigma**2)/(sqrt2pi*sigma))
    return obsl

@njit
def log_obs_like(X,Y,Nvar,sigma):
    """
    Observation likelyhood from data
    """
    log_obsl = -0.5*np.sum(np.log(sigma)) - (Nvar/2.0)*np.log(2*np.pi) -0.5*np.sum(((X-Y)/sigma)**2)
    return log_obsl

@njit
def MCMC(M,n_estim,param_0,L,proposal_kernel,prior,prior_pars,mean_kern,cov_kern,estimate_L,data,R,model,obs_li_f,obs_li_param,h,sqh,known_param):
    parameters = np.zeros((M+1,n_estim)) 
    parameters[0] = param_0
    val_prior = prior(param_0,prior_pars)
    naccept = 0
    for m in range(1,M+1): #montecarlo steps
        valid = False
        while not(valid):
            proposal = parameters[m-1] + proposal_kernel(mean_kern,cov_kern,n_estim)
            val_prior_prop = prior(proposal,prior_pars)
            if val_prior_prop > 0 : valid = True
        L_star = estimate_L(data,R,model,proposal,obs_li_f,obs_li_param,h,sqh,known_param) #For this implementation, pseudoespectral method with a bootstrap particle filter.
        alpha = np.min(np.array([1,(L_star*val_prior_prop)/(L*val_prior)]))
        u = np.random.uniform(0,1)
        if u < alpha: 
            parameters[m] = proposal
            L = L_star
            val_prior = val_prior_prop
            naccept += 1            
        else:
            parameters[m] = parameters[m-1]
        if m%int(M*0.05) == 0: 
            print("m=",m,"rate accepted","{:.3f}".format(naccept/m),"total accepted",naccept)
            print("Current pars.","{:.3f},{:.3f}".format(parameters[m]))
            print("Current Like","{:.3f}".format(L))
            print("Proposal pars.","{:.3f},{:.3f}".format(proposal))
            print("Proposal Like","{:.3f}".format(L_star))
            print("u","{:.5f}".format(u),"alpha","{:.5f}".format(alpha))
    return parameters, naccept, L


@njit
def ln_MCMC(M,n_estim,param_0,ln_L,proposal_kernel,prior,prior_pars,mean_kern,cov_kern,estimate_ln_L,data,R,model,obs_li_f,obs_li_param,h,sqh,known_param):
    parameters = np.zeros((M+1,n_estim)) 
    parameters[0] = param_0
    ln_val_prior = np.log(prior(param_0,prior_pars))
    naccept = 0
    accepted = np.zeros(M+1)
    v_ln_Ls = np.zeros(M+1)
    v_ln_Ls[0] = ln_L
    for m in range(1,M+1): #montecarlo steps
        valid = False
        while not(valid):
            proposal = parameters[m-1] + proposal_kernel(mean_kern,cov_kern,n_estim)
            val_prior_prop = prior(proposal,prior_pars)
            if val_prior_prop > 0 : valid = True
        ln_val_prior_prop = np.log(val_prior_prop)
        ln_L_star = estimate_ln_L(data,R,model,proposal,obs_li_f,obs_li_param,h,sqh,known_param)
        aa = ln_L_star + ln_val_prior_prop - ln_L - ln_val_prior
        ln_alpha = np.min(np.array([0.0, aa]))
        alpha = np.exp(ln_alpha)
        u = np.random.uniform(0,1)
        if u < alpha: 
            parameters[m] = proposal
            ln_L = ln_L_star
            ln_val_prior = ln_val_prior_prop
            naccept += 1
        else:
            parameters[m] = parameters[m-1]
        accepted[m] = naccept
        v_ln_Ls[m] = ln_L
        if m%int(M*0.05) == 0: 
            print("m=",m,"rate accepted",naccept/m,"total accepted",naccept,"--"*10)
            print("Current Like",ln_L,ln_val_prior)
            print("Current par.",parameters[m])
            print("Proposal Like",ln_L_star,ln_val_prior_prop)
            print("Proposal pars.",proposal)
            print("u",u,"alpha",alpha,"aa",aa)
            print("ln_L var", np.var(v_ln_Ls[:m]))
            print("--"*20)
    return parameters, accepted, v_ln_Ls


@njit(parallel=True)
def bootstrap(data,R,model,proposal,obs_li_f,obs_li_param,h,sqh,known_param):
    ncoord_t,N = np.shape(data) #number of coordenates and time observations
    Ncoord = ncoord_t-1 #take out the time.
    state = getIC(data[:Ncoord],R) #initialize the R particles. 
    ts = data[Ncoord] #get the data points.
    weights = np.zeros(R) #initialize the weights
    Lstar = 1
    for i in range(N-1): #loop over the observaitons
        t = ts[i]
        Dt = ts[i+1] - t #step beteewn observations
        Nt = int(Dt/h) #steps to perform between the timecourse data and the h steps of the simulation
        Y = data[:Ncoord,i+1]  #data points
        for k in prange(R): #loop over the particles
            state[k],t_ev = model(state[k],t,h,sqh,Nt,known_param,proposal) #evolve from t_i to t_i+1
            w = obs_li_f(state[k],Y,Ncoord,obs_li_param)
            weights[k] = w
        wsum = np.sum(weights)
        if wsum == 0: 
            Lstar=0
            break
        Lstar = Lstar*wsum/R #TODO: make it log once this works.
        w_norm = weights/wsum
        cum_w = np.cumsum(w_norm)
        state_resampled = np.zeros((R,Ncoord))
        for k in prange(R):
            state_resampled[k] = state[np.where(cum_w>=np.random.uniform(0,1))[0][0]]
        state = state_resampled
    return Lstar


@njit(parallel=True)
def ln_bootstrap(data,R,model,proposal,log_obs_li_f,obs_li_param,h,sqh,known_param):
    ncoord_t,N = np.shape(data) #number of coordenates and time observations
    Ncoord = ncoord_t-1 #take out the time.
    state = getIC(data[:Ncoord],R) #initialize the R particles. 
    ts = data[Ncoord] #get the data points.
    weights = np.zeros(R,dtype=np.float64) #initialize the weights
    ln_weights = np.zeros(R,dtype=np.float64) #initialize the weights
    ln_Lstar = 0
    for i in range(N-1): #loop over the observaitons
        t = ts[i]
        Dt = ts[i+1] - t #step beteewn observations
        Nt = int(Dt/h) #steps to perform between the timecourse data and the h steps of the simulation
        Y = data[:Ncoord,i+1]  #data points
        for k in prange(R): #loop over the particles
            state[k],t_ev = model(state[k],t,h,sqh,Nt,known_param,proposal) #evolve from t_i to t_i+1
            lnw =  log_obs_li_f(state[k],Y,Ncoord,obs_li_param)
            ln_weights[k] = lnw
        weights = np.exp(ln_weights)
        marginal = np.sum(weights)
        if marginal == 0: 
            ln_Lstar = -np.inf
            break
        ln_Lstar = ln_Lstar + np.log(marginal)  
        w_norm = weights/marginal
        cum_w = np.cumsum(w_norm)
        state_resampled = np.zeros((R,Ncoord))
        for k in prange(R):
            state_resampled[k] = state[np.where(cum_w>=np.random.uniform(0,1))[0][0]]
        state = state_resampled
    ln_Lstar = ln_Lstar - (N-1)*np.log(R)
    return ln_Lstar


def convergence(chains,nparam,R,M):
    combined = np.zeros((nparam,R*M))
    for i in range(R):
        for p in range(nparam):
            combined[p,i*M:M*(i+1)] = chains[i][p]
    ranks = np.zeros((nparam,R*M)) 
    for p in range(nparam):
        ranks[p] = np.argsort(combined[p]) + 1 #Python starts the indexing at 0, need to start at 1.
    zetas = np.zeros((nparam,R,M))
    for p in range(nparam):
        for i in range(R):
            zetas[p][i] = ndtri((ranks[p,i*M:M*(i+1)]-1/2)/(R*M)) 
    zkpl,zkmi = np.zeros((nparam,R)),np.zeros((nparam,R))
    zk = np.zeros(nparam)
    skpl,skmi = np.zeros((nparam,R)),np.zeros((nparam,R))
    for p in range(nparam):
        for i in range(R):
            zkpl[p][i] = 2/M*np.sum(zetas[p,i,int(M/2):])
            zkmi[p][i] = 2/M*np.sum(zetas[p,i,:int(M/2)])
            skpl[p][i] = 2/(M-2)*np.sum(np.pow(zetas[p,i,int(M/2):]-zkpl[p][i],2))
            skmi[p][i] = 2/(M-2)*np.sum(np.pow(zetas[:p,i,int(M/2)]-zkmi[p][i],2))
        zk[p] = 0.5/R*np.sum(zkpl[p]+zkmi[p])
    B,W = np.zeros(nparam),np.zeros(nparam) 
    for p in range(nparam):
        B[p] = M/(4*R-2)*np.sum(np.pow(zkpl[p] - zk[p],2)+np.pow(zkmi[p] - zk[p],2))
        W[p] = 0.5/R*np.sum(skpl[p]+skmi[p])
    hatR,V = np.zeros(nparam),np.zeros(nparam)
    for p in range(nparam):
        V[p] = (M-2)/M*W[p] + 2/M*B[p]
        hatR[p] = np.sqrt(V[p]/W[p])
    ESS = Seff(chains,nparam,R,M,W,V)
    return hatR,ESS


def Seff(chains,nparam,R,M,W,varhat):
    rho = np.zeros((nparam,R,M))
    tau = np.zeros(nparam)
    for k in range(nparam): #loop over parameters
        for j in range(R): #loop over chains
            rho[k][j] = autocorrelation(chains[j][k],M)
    for k in range(nparam):
        rhohat = np.zeros(M)
        for j in range(R): #loop over chains
            rhohat = rhohat + rho[k,j,:]
        rhohat = rhohat/R
        rhohat = W[k] - rhohat
        rhohat = rhohat/varhat[k]
        rhohat = 1.0 - rhohat
        T = 0
        while T < M - 3 and rhohat[T+1] + rhohat[T+2] > 0:
            T = T+2
        tau[k] = 1+2*np.sum(rhohat[:T])
    return (M*nparam)/tau


def autocorrelation(data,max_lag):
    """
    Compute the autocorrelation function for the input array `x` up to `max_lag`.
    """
    n = len(data)
    result = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[-n:]
    result /= result[0]  # Normalize
    return result[:max_lag+1]

