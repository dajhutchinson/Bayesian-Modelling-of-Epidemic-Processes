import Models,ABC

import numpy as np

from sklearn.linear_model import LinearRegression
from itertools import combinations
from random import randint
from sys import maxsize
from scipy import special,stats

"""
    Approximately Sufficient Subset
"""
def joyce_marjoram(summary_stats:["function"],n_obs:int,y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],param_bounds:[(float,float)],
    distance_measure=ABC.l2_norm,KERNEL=ABC.uniform_kernel,BANDWIDTH=1,n_samples=10000,n_bins=10,printing=True) -> [int]:
    """
    DESCRIPTION
    Use the algorithm in Paul Joyce, Paul Marjoram 2008 to find an approxiamtely sufficient set of summary statistics (from set `summary_stats`)

    PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. These are what will be evaluated
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    param_bounds ([(float,float)]) - The bounds of the priors used to generate parameter sets.
    KERNEL (func) - one of the kernels defined above. determine which parameters are good or not.
    BANDWIDTH (float) - scale parameter for `KERNEL`
    n_samples (int) - number of samples to make
    n_bins (int) - Number of bins to discretise each dimension of posterior into (default=10)

    RETURNS
    [int] - indexes of selected summary stats in `summary_stats`
    """

    if (type(y_obs)!=list): raise TypeError("`y_obs` must be a list (not {})".format(type(y_obs)))
    if (len(y_obs)!=n_obs): raise ValueError("Wrong number of observations supplied (len(y_obs)!=n_obs) ({}!={})".format(len(y_obs),n_obs))
    if (len(priors)!=fitting_model.n_params): raise ValueError("Wrong number of priors given (exp fitting_model.n_params={})".format(fitting_model.n_params))

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # generate samples
    SAMPLES=[] # (theta,s_vals)
    for i in range(n_samples):
        if (printing): print("{:,}/{:,}".format(i+1,n_samples),end="\r")

        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in priors]

        # observe theorised model
        fitting_model.update_params(theta_t)
        y_t=fitting_model.observe()
        s_t=[s(y_t) for s in summary_stats]

        SAMPLES.append((theta_t,s_t))

    if (printing):
        print()
        for i in range(len(summary_stats)):
            print("var_{}={:,.3f}".format(i,np.var([x[1][i] for x in SAMPLES])))

    # consider adding each summary stat in turn
    ACCEPTED_SUMMARY_STATS_ID=[] # index of accepted summary stats

    id_to_try=randint(0,len(summary_stats)-1)
    ACCEPTED_SUMMARY_STATS_ID=[id_to_try]
    tried=[]

    while True:
        if (printing): print("Currently accepted - ",ACCEPTED_SUMMARY_STATS_ID)

        # samples using current accepted summary stats
        samples_curr=[(theta,[s[j] for j in ACCEPTED_SUMMARY_STATS_ID]) for (theta,s) in SAMPLES]
        s_obs_curr=[s_obs[j] for j in ACCEPTED_SUMMARY_STATS_ID]
        accepted_params_curr=[]
        for (theta_t,s_t) in samples_curr:
            norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs_curr)]
            if (KERNEL(ABC.l1_norm(norm_vals),BANDWIDTH)): # NOTE - ABC.l1_norm() can be replaced by anyother other norm
                accepted_params_curr.append(theta_t)

        # chooose next ss to try
        available_ss=[x for x in range(len(summary_stats)-len(tried)) if (x not in ACCEPTED_SUMMARY_STATS_ID) and (x not in tried)]
        if (len(available_ss)==0): return ACCEPTED_SUMMARY_STATS_ID

        id_to_try=available_ss[randint(0,len(available_ss)-1)]
        tried+=[id_to_try]
        if (printing): print("Trying to add {} to [{}]".format(id_to_try,",".join([str(x) for x in ACCEPTED_SUMMARY_STATS_ID])))

        # samples using current accepted summary stats and id_to_try
        samples_prop=[(theta,[s[j] for j in ACCEPTED_SUMMARY_STATS_ID+[id_to_try]]) for (theta,s) in SAMPLES]
        s_obs_prop=[s_obs[j] for j in ACCEPTED_SUMMARY_STATS_ID+[id_to_try]]
        accepted_params_prop=[]
        for (theta_t,s_t) in samples_prop:
            norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs_prop)]
            if (KERNEL(ABC.l1_norm(norm_vals),BANDWIDTH)): # NOTE - ABC.l1_norm() can be replaced by anyother other norm
                accepted_params_prop.append(theta_t)


        if (printing): print("N_(k-1)={:,}".format(len(accepted_params_curr)))
        if (printing): print("N_k    ={:,}".format(len(accepted_params_prop)))
        if (__compare_summary_stats(accepted_params_curr,accepted_params_prop,param_bounds,n_params=len(priors),n_bins=10)):
            # add id_to_try
            ACCEPTED_SUMMARY_STATS_ID+=[id_to_try]
            if (printing): print("Accepting {}.\nCurrently accepted - ".format(id_to_try),ACCEPTED_SUMMARY_STATS_ID)

            # consider removing previous summaries
            if (printing): print("\nConsider removing previous summaries")
            for i in range(len(ACCEPTED_SUMMARY_STATS_ID)-2,-1,-1):
                ids_minus=[x for (j,x) in enumerate(ACCEPTED_SUMMARY_STATS_ID) if j!=i]
                if (printing): print("Comparing [{}] to [{}]".format(",".join([str(x) for x in ACCEPTED_SUMMARY_STATS_ID]),",".join([str(x) for x in ids_minus])))

                # samples using reduced set
                samples_minus=[(theta,[s[j] for j in ids_minus]) for (theta,s) in SAMPLES]
                s_obs_minus=[s_obs[j] for j in ids_minus]
                accepted_params_minus=[]
                for (theta_t,s_t) in samples_minus:
                    norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs_minus)]
                    if (KERNEL(ABC.l1_norm(norm_vals),BANDWIDTH)): # NOTE - ABC.l1_norm() can be replaced by anyother other norm
                        accepted_params_minus.append(theta_t)

                if (__compare_summary_stats(accepted_params_prop,accepted_params_minus,param_bounds,n_params=len(priors),n_bins=10)):
                    if (printing): print("Removing - ",ACCEPTED_SUMMARY_STATS_ID[i])
                    ACCEPTED_SUMMARY_STATS_ID=ids_minus

            if (printing): print("Reduced to - ",ACCEPTED_SUMMARY_STATS_ID)


        if (printing): print()

    return ACCEPTED_SUMMARY_STATS_ID

def __compare_summary_stats(accepted_params_curr:[[float]],accepted_params_prop:[[float]],param_bounds:[(float,float)],n_params:int,n_bins=10) -> bool:
    """
    DESCRIPTION
    The algorithm proposed by joyce-marjoram for estimating the odds-ratio for two sets of summary statistics S_{K-1} and S_K
    where S_K is a super-set of S_{K-1}

    PARAMETERS
    accepted_params_curr ([[float]]) - Sets of parameters which were accepted when using current set of summary stats S_{K-1}.
    accepted_params_prop ([[float]]) - Sets of parameters which were accepted when using propsed set of summary stats S_K.
    param_bounds ([(float,float)]) - The bounds of the priors used to generate parameter sets.
    n_params (int) - Number of parameters being fitted.
    n_bins (int) - Number of bins to discretise each dimension of posterior into (default=10)

    RETURNS
    bool - Whether S_K produces a notably different posterior to S_{K-1} (ie whether to accept new summary stat or not)
    """
    n_curr=len(accepted_params_curr)
    if (n_curr==0): return True # if nothing is currently being used then accept
    n_prop=len(accepted_params_prop)

    # count occurences of accepted params
    bins_curr=[[0 for _ in range(n_bins)] for _ in range(n_params)]
    for params in accepted_params_curr:
        for (dim,param) in enumerate(params):
            step=(param_bounds[dim][1]-param_bounds[dim][0])/(n_bins-1)
            if (step==0): step=1 # avoid division my zero

            i=int(np.floor((param-param_bounds[dim][0])/step))
            bins_curr[dim][i]+=1

    bins_prop=[[0 for _ in range(n_bins)] for _ in range(n_params)]
    for params in accepted_params_prop:
        for (dim,param) in enumerate(params):
            step=(param_bounds[dim][1]-param_bounds[dim][0])/(n_bins-1)
            if (step==0): step=1 # avoid division my zero

            i=int(np.floor((param-param_bounds[dim][0])/step))
            bins_prop[dim][i]+=1

    # calculated expected number of occurences for each bin
    expected=[[(x*n_prop)/n_curr for x in bins] for bins in bins_curr]
    sd=[[np.sqrt(expected[i][j]*((n_curr-x)/n_curr)) for (j,x) in enumerate(bins)] for (i,bins) in enumerate(bins_curr)]

    upper_thresh=[[expected[i][j]+4*sd[i][j] for j in range(n_bins)] for i in range(n_params)]
    lower_thresh=[[expected[i][j]-4*sd[i][j] for j in range(n_bins)] for i in range(n_params)]

    # count number of extreme values
    # value is extreme if it is more than 4sd away from expected
    extreme=0
    for i in range(n_params):
        for j in range(n_bins):
            if (bins_prop[i][j]>upper_thresh[i][j]) or (bins_prop[i][j]<lower_thresh[i][j]): extreme+=1

    return (extreme>0)

"""
    MINIMUM ENTROPY
"""
def minimum_entropy(summary_stats:["function"],n_obs:int,y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],min_subset_size=1,max_subset_size=None,n_samples=1000,n_accept=100,k=4,printing=False) -> ([int],[[float]]):
    """

    RETURNS
    [int] - indexes of best summary stats
    [[float]] - list of all accepted theta when "best summary stats"
    """

    lowest=([],maxsize,[])

    # all permutations of summary stats
    n_stats=len(summary_stats)
    max_subset_size=max_subset_size if (max_subset_size) else n_stats
    perms=[]
    for n in range(max(min_subset_size,1),min(n_stats+1,max_subset_size+1)):
        perms+=[x for x in combinations([i for i in range(n_stats)],n)]

    sampling_details={"sampling_method":"best","num_runs":n_samples,"sample_size":n_accept}

    for (j,perm) in enumerate(perms):
        if (printing): print("Permutation = ",perm,sep="")
        else: print("({}/{})".format(j,len(perms)),end="\r")
        ss=[summary_stats[i] for i in perm]
        _,accepted_theta=ABC.abc_rejection(n_obs,y_obs,fitting_model,priors,sampling_details,summary_stats=ss,show_plots=False,printing=printing)

        estimate_ent=__k_nn_estimate_entropy(len(priors),accepted_theta,k=k)
        if (printing): print("Estimate_ent of ",perm,"= {:,.2f}\n".format(estimate_ent),sep="")
        if (estimate_ent<lowest[1]): lowest=(perm,estimate_ent,accepted_theta)

    # return lowest[1]
    return lowest[0],lowest[2]

def __k_nn_estimate_entropy(n_params:int,parameter_samples:[(float)],k=4) -> float:
    """
    DESCRIPTION
    Kth Nearest Neighbour estimate of entropy for a posterior distribution.

    PARAMETERS
    n_params (int) - Number of parameters being fitted.
    parameter_samples ([(float)]) - Set of accepted sampled parameters.

    OPTIONAL PARAMETERS
    k (int) - Which nearest neighbour to consider (default=4)

    RETURNS
    float - estimated entropy
    """
    n=len(parameter_samples) # number accepted samples
    if (k>n): raise ValueError("k cannot be greater than the number of samples")

    gamma=special.gamma(1+n_params/2)
    digamma=special.digamma(k)

    h_hat=np.log(np.pi**(n_params/2)/gamma)
    h_hat-=digamma
    h_hat+=np.log(n)

    constant=n_params/n
    for i in range(n):
        sample_i=parameter_samples[i]
        distances=[]
        for j in range(n): # find kth nearest neighbour
            if (j==i): continue
            sample_j=parameter_samples[j]
            distances.append(ABC.l2_norm(sample_i,sample_j))
        distances.sort()
        h_hat+=constant*np.log(distances[3])

    return h_hat

"""
    TWO-STEP MINIMUM ENTROPY
"""
def two_step_minimum_entropy(summary_stats:["function"],n_obs:int,y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],min_subset_size=1,max_subset_size=None,n_samples=1000,n_accept=100,n_keep=10,k=4,printing=False) -> ([int],[[float]]):
    """
    OPTIONAL PARAMETERS
    n_keep (int) - number of (best) accepted samples to keep from the set of stats which minimise entropy (`best_stats`) and use for evaluating second stage (default=10)
    """
    n_stats=len(summary_stats)
    max_subset_size=max_subset_size if (max_subset_size) else n_stats

    # find summary stats which minimise entropy
    me_stats_id,accepted_theta=minimum_entropy(summary_stats,n_obs,y_obs,fitting_model,priors,min_subset_size=min_subset_size,max_subset_size=max_subset_size,n_samples=n_samples,n_accept=n_accept,k=k,printing=printing)
    me_stats=[summary_stats[i] for i in me_stats_id]
    s_obs=[s(y_obs) for s in me_stats]
    if (printing): print("ME stats found -",me_stats_id,"\n")

    # identify the `n_keep` best set of parameters
    theta_scores=[]
    for (i,theta) in enumerate(accepted_theta):

        fitting_model.update_params(theta)
        y_t=fitting_model.observe()
        s_t=[s(y_t) for s in me_stats]

        weight=ABC.l1_norm([ABC.l2_norm(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)])
        theta_scores.append((weight,i))

    theta_scores.sort(key=lambda x:x[0])
    me_theta=[accepted_theta[x[1]] for x in theta_scores[:n_keep]]
    if (printing): print("ME theta found.\n")

    # all permutations of summary stats
    n_stats=len(summary_stats)
    perms=[]
    for n in range(min_subset_size,max_subset_size+1):
        perms+=[x for x in combinations([i for i in range(n_stats)],n)]

    lowest=([],maxsize,[])

    # compare subsets of summary stats to
    sampling_details={"sampling_method":"best","num_runs":n_samples,"sample_size":n_accept,"distance_measure":ABC.log_l2_norm}

    for (i,perm) in enumerate(perms):
        if (printing): print("Permutation = ",perm,sep="")
        else: print("{}/{}           ".format(i,len(perms)),end="\r")
        ss=[summary_stats[i] for i in perm]
        _,accepted_theta=ABC.abc_rejection(n_obs,y_obs,fitting_model,priors,sampling_details,summary_stats=ss,show_plots=False,printing=printing)

        rsses=[__rsse(accepted_theta,theta) for theta in me_theta]
        mrsse=np.mean(rsses)
        if (printing): print("MRSSE of ",perm,"= {:,.2f}\n".format(mrsse),sep="")
        if (mrsse<lowest[1]): lowest=(perm,mrsse,accepted_theta)

    return lowest[0],lowest[2]

def __rsse(obs,target) -> float:
    # Residual Sum of Squares Error

    error=sum([ABC.l2_norm(o,target)**2 for o in obs])
    error=np.sqrt(error)
    error/=len(obs)

    return error

"""
    SEMI-AUTO ABC
"""
def abc_semi_auto(n_obs:int,y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],distance_measure=ABC.l2_norm,n_pilot_samples=10000,n_pilot_acc=1000,n_params_sample_size=100,summary_stats=None,printing=True) -> (["function"],[[float]]):

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])

    sampling_details={"sampling_method":"best","num_runs":n_pilot_samples,"sample_size":n_pilot_acc,"distance_measure":distance_measure,"params_sample_size":n_params_sample_size}

    #perform pilot run
    _,pilot_params=ABC.abc_rejection(n_obs=n_obs,y_obs=y_obs,fitting_model=fitting_model,priors=priors,sampling_details=sampling_details,summary_stats=summary_stats,show_plots=False,printing=printing)

    # calculate distribution of accepted params
    new_priors=[]
    for i in range(fitting_model.n_params):
        pilot_params_dim=[x[i] for x in pilot_params]
        dist=stats.gaussian_kde(pilot_params_dim)
        new_priors.append(dist)
    if (printing): print("Calculated posteriors from pilot.")

    # Sample new parameters and simulate model
    m=sampling_details["params_sample_size"] if ("params_sample_size" in sampling_details) else 1000

    samples=[]
    for i in range(m):
        if (printing): print("{}/{}".format(i,m),end="\r")
        theta_t=[list(p.resample(1))[0][0] for p in new_priors]

        # observe theorised model
        fitting_model.update_params(theta_t)
        y_t=fitting_model.observe()
        s_t=[s(y_t) for s in summary_stats]

        samples.append((theta_t,s_t))
    if (printing): print("Generated {} parameter sets.".format(m))

    # create summary stats
    # NOTE - other methods can be used
    new_summary_stats=[]
    X=[list(np.ravel(np.matrix(x[1]))) for x in samples] # flatten output data
    X=np.array(X)
    coefs=[]

    for i in range(fitting_model.n_params):
        y=np.array([x[0][i] for x in samples])

        reg=LinearRegression().fit(X, y)
        coefs.append(list(reg.coef_))

    new_summary_stats=[lambda xs: list(np.dot(coefs,np.ravel(np.matrix(xs))))]
    s_t=[s(samples[0][1]) for s in new_summary_stats]
    if (printing): print("Generated summary statistics")

    return new_summary_stats,coefs
