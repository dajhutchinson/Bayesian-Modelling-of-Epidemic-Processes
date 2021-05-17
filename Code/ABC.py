import Models,Plotting

from scipy import stats
from sys import maxsize

import numpy as np
import matplotlib.pyplot as plt

"""
    KERNELS
"""
def abstract_kernel(x:float,epsilon:float) -> bool:
    """
    DESCRIPTION
    determine whether to accept an observation based on how far it is from other observations.

    PARAMETERS
    x (float) - value for comparision (typically distance between two observations)
    epsilon (float) - scaling factor

    RETURNS
    bool - whether to accept
    """
    return True

def uniform_kernel(x:float,epsilon:float) -> bool:
    return abs(x)<=epsilon

def epanechnikov_kernel(x:float,epsilon:float) -> bool:
    if (abs(x)>epsilon): return False
    ep_val=(1-(x/epsilon)**2)#*(1/epsilon)*(3/4) # prob to accept
    x=stats.uniform(0,1).rvs(1)[0] # sample from U[0,1]
    return (x<=ep_val)

def gaussian_kernel(x:float,epsilon:float) -> bool:
    gaus_val=np.exp(-(1/2)*((x*epsilon)**2)) #*(1/np.sqrt(2*np.pi*(epsilon**2))) # prob to accept
    x=stats.uniform(0,1).rvs(1)[0] # sample from U[0,1]
    return (x<=gaus_val)

"""
    DISTANCE MEASURES
"""
def l1_norm(xs:[float],ys=[]) -> float:
    return sum(xs)+sum(ys)

def l2_norm(s_t:(float),s_obs:(float)) -> float:
    return sum([(x-y)**2 for (x,y) in zip(s_t,s_obs)])**.5

def log_l2_norm(s_t:(float),s_obs:(float)) -> float:

    return sum([(np.log(x)-np.log(y))**2 if (x>0 and y>0) else 0 for (x,y) in zip(s_t,s_obs)])**.5

def l_infty_norm(xs:[float]) -> float:
    return max(xs)

"""
    REJECTION SAMPLING METHODS
"""

def __sampling_stage_fixed_number(DESIRED_SAMPLE_SIZE:int,EPSILON:float,KERNEL:"func",PRIORS:["stats.Distribution"],
    s_obs:[float],model_t:Models.Model,summary_stats:["function"],distance_measure=l2_norm,printing=True) -> ([[float]],[[float]],[[float]]):
    """
    DESCRIPTION
    keep generating parameter values and observing the equiv models until a sufficient number of `good` parameter values have been found.

    PARAMETERS
    DESIRED_SAMPLE_SIZE (int) - number of `good` samples to wait until
    EPSILON (int) - scale parameter for `KERNEL`
    KERNEL (func) - one of the kernels defined above. determine which parameters are good or not.
    PRIORS ([stats.Distribution]) - prior distribution for parameter values (one per parameter)
    s_obs ([float]) - summary statistic values for observations from true model.
    summary_stat ([func]) - functions used to determine each summary statistic.

    OPTIONAL PARAMETERS
    distance_measure (func) - distance function to use (See choices above)
    printing (bool) - whether to print updates to terminal (default=True)

    RETURNS
    [[float]] - accepted parameter values
    [[float]] - observations made when using accepted parameter values
    [[float]] - summary statistic values for accepted parameter observations
    """

    ACCEPTED_PARAMS=[]
    ACCEPTED_OBS=[]
    ACCEPTED_SUMMARY_VALS=[]
    i=0
    while (len(ACCEPTED_OBS)<DESIRED_SAMPLE_SIZE):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS]

        # observe theorised model
        model_t.update_params(theta_t)
        y_t=model_t.observe()
        s_t=[s(y_t) for s in summary_stats]

        # accept-reject
        norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)]
        if (KERNEL(l1_norm(norm_vals),EPSILON)): # NOTE - l1_norm() can be replaced by anyother other norm
            ACCEPTED_PARAMS.append(theta_t)
            ACCEPTED_OBS.append(y_t)
            ACCEPTED_SUMMARY_VALS.append(s_t)

        i+=1
        if (printing): print("({:,}) {:,}/{:,}".format(i,len(ACCEPTED_PARAMS),DESIRED_SAMPLE_SIZE),end="\r") # update user on sampling process
    if (printing): print("\n")

    return ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS

def __sampling_stage_best_samples(NUM_RUNS:int,SAMPLE_SIZE:int,PRIORS:["stats.Distribution"],
    s_obs:[float],model_t:Models.Model,summary_stats:["function"],distance_measure=l2_norm,printing=True) -> ([[float]],[[float]],[[float]]):
    """
    DESCRIPTION
    perform `NUM_RUNS` samples and return the parameters values associated to the best `SAMPLE_SIZE`.

    PARAMETERS
    NUM_RUNS (int) - number of samples to make
    SAMPLE_SIZE (int) - The best n set of parameters to return
    PRIORS ([stats.Distribution]) - prior distribution for parameter values (one per parameter)
    s_obs ([float]) - summary statistic values for observations from true model.
    summary_stat ([func]) - functions used to determine each summary statistic.

    OPTIONAL PARAMETERS
    distance_measure - (func) - distance function to use (See choices above)
    printing (bool) - whether to print updates to terminal (default=True)

    RETURNS
    [[float]] - accepted parameter values
    [[float]] - observations made when using accepted parameter values
    [[float]] - summary statistic values for accepted parameter observations
    """

    SAMPLES=[None for _ in range(SAMPLE_SIZE)]

    for i in range(NUM_RUNS):
        # sample parameters
        theta_t=[pi_i.rvs(1)[0] for pi_i in PRIORS]

        # observe theorised model
        model_t.update_params(theta_t)
        y_t=model_t.observe()
        s_t=[s(y_t) for s in summary_stats]

        norm_vals=[distance_measure(s_t_i,s_obs_i) for (s_t_i,s_obs_i) in zip(s_t,s_obs)]
        summarised_norm_val=l1_norm(norm_vals) # l1_norm can be replaced by any other norm

        for j in range(max(0,SAMPLE_SIZE-i-1,SAMPLE_SIZE)):
            # truncated insertion sort
            if (SAMPLES[j]==None) or (SAMPLES[j][0]>summarised_norm_val):
                if (j>0): SAMPLES[j-1]=SAMPLES[j]
                SAMPLES[j]=(summarised_norm_val,(theta_t,y_t,s_t))

        if (printing): print("({:,}/{:,})".format(i,NUM_RUNS),end="\r") # update user on sampling process

    if (printing): print("                                                 ",end="\r")
    SAMPLES=[x[1] for x in SAMPLES] # remove norm value
    ACCEPTED_PARAMS=[x[0] for x in SAMPLES]
    ACCEPTED_OBS=[x[1] for x in SAMPLES]
    ACCEPTED_SUMMARY_VALS=[x[2] for x in SAMPLES]

    return ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS


"""
    ABC-Rejection Sampling
"""

def abc_rejection(n_obs:int,y_obs:[[float]],fitting_model:Models.Model,priors:["stats.Distribution"],sampling_details:dict,summary_stats=None,show_plots=True,printing=True) -> (Models.Model,[[float]]):
    """
    DESCRIPTION
    Rejction Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Models.Model) - Models.Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    sampling_details - specification of how sampling should be done (see README.md)

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)
    show_plots (bool) - whether to generate and show plots (default=True)
    printing (bool) - whether to print updates to terminal (default=True)

    RETURNS
    Models.Model - fitted model with best parameters
    [[float]] - set of all accepted parameter values (use for further investigation)
    """

    if (type(y_obs)!=list): raise TypeError("`y_obs` must be a list (not {})".format(type(y_obs)))
    if (len(y_obs)!=n_obs): raise ValueError("Wrong number of observations supplied (len(y_obs)!=n_obs) ({}!={})".format(len(y_obs),n_obs))
    if (len(priors)!=fitting_model.n_params): raise ValueError("Wrong number of priors given (exp fitting_model.n_params={})".format(fitting_model.n_params))

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # sampling step
    if ("sampling_method" not in sampling_details): raise Exception("Insufficient details provided in `sampling_details` - missing `sampling_method`")

    elif (sampling_details["sampling_method"]=="fixed_number"):
        if any([x not in sampling_details for x in ["sample_size","scaling_factor","kernel_func"]]): raise Exception("`sampling_details` missing key(s) - expecting `sample_size`,`scaling_factor` and `kernel_func`")
        sample_size=sampling_details["sample_size"]
        epsilon=sampling_details["scaling_factor"]
        kernel=sampling_details["kernel_func"]
        distance_measure=l2_norm if (not "distance_measure" in sampling_details) else sampling_details["distance_measure"]
        ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS=__sampling_stage_fixed_number(sample_size,epsilon,kernel,PRIORS=priors,s_obs=s_obs,model_t=fitting_model,summary_stats=summary_stats,distance_measure=distance_measure,printing=printing)

    elif (sampling_details["sampling_method"]=="best"):
        if any([x not in sampling_details for x in ["sample_size","num_runs"]]): raise Exception("`sampling_details` missing key(s) - expecting `num_runs` and `sample_size`")
        num_runs=sampling_details["num_runs"]
        sample_size=sampling_details["sample_size"]
        distance_measure=l2_norm if (not "distance_measure" in sampling_details) else sampling_details["distance_measure"]
        ACCEPTED_PARAMS,ACCEPTED_OBS,ACCEPTED_SUMMARY_VALS=__sampling_stage_best_samples(num_runs,sample_size,PRIORS=priors,s_obs=s_obs,model_t=fitting_model,summary_stats=summary_stats,distance_measure=distance_measure,printing=printing)

    # best estimate of model
    theta_hat=[np.mean([p[i] for p in ACCEPTED_PARAMS]) for i in range(fitting_model.n_params)]
    model_hat=fitting_model.copy(theta_hat)
    s_hat=[s(model_hat.observe()) for s in summary_stats]

    if (show_plots):
        fig=plt.figure(constrained_layout=True)
        fig=__abc_rejection_plotting(fig,y_obs,priors,fitting_model,model_hat,ACCEPTED_SUMMARY_VALS,ACCEPTED_OBS,ACCEPTED_PARAMS)
        # plt.get_current_fig_manager().window.state("zoomed")
        plt.show()

    return model_hat,ACCEPTED_PARAMS

def __abc_rejection_plotting(fig:plt.Figure,y_obs:[[float]],priors:["stats.Distribution"],fitting_model:Models.Model,model_hat:Models.Model,
                            accepted_summary_vals:[[float]],accepted_observations:[[float]],accepted_params:[[float]]) -> plt.Figure:
    # plot results
    n_simple_ss=sum(len(s)==1 for s in accepted_summary_vals[0]) # number of summary stats which map to a single dimension
    n_cols=2 if (n_simple_ss==0) else 3
    n_params=(fitting_model.n_params-2) if (type(fitting_model) is Models.SIRModel) else fitting_model.n_params
    n_rows=max([1,np.lcm.reduce([n_params,max(1,n_simple_ss),fitting_model.dim_obs])])

    # plot accepted obervations for each dimension
    gs=fig.add_gridspec(n_rows,n_cols)

    # plot accepted observations (each dimension separate)
    row_step=n_rows//fitting_model.dim_obs
    for i in range(fitting_model.dim_obs):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        y_obs_dim=[y[i] for y in y_obs]
        accepted_obs_dim=[[y[i] for y in obs] for obs in accepted_observations]
        Plotting.plot_accepted_observations(ax,fitting_model.x_obs,y_obs_dim,accepted_obs_dim,model_hat,dim=i)

    # plot posterior for each parameter
    row_step=n_rows//n_params
    if (type(fitting_model) is Models.SIRModel):
        for i in range(2,fitting_model.n_params):
            name="Theta_{}".format(i)
            ax=fig.add_subplot(gs[(i-2)*row_step:(i-2+1)*row_step,0])
            accepted_vals=[x[i] for x in accepted_params]
            Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=model_hat.params[i],prior=priors[i],dim=i)
    else:
        for i in range(fitting_model.n_params):
            name="Theta_{}".format(i)
            ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
            accepted_vals=[x[i] for x in accepted_params]
            Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=model_hat.params[i],prior=priors[i],dim=i)

    # plot histogram of each summary statistic value
    row=0
    if (n_simple_ss!=0):
        row_step=n_rows//n_simple_ss
        for i in range(len(summary_stats)):
            if (len(accepted_summary_vals[0][i])==1):
                name="s_{}".format(i)
                ax=fig.add_subplot(gs[row*row_step:(row+1)*row_step,1])
                row+=1
                accepted_vals=[s[i][0] for s in accepted_summary_vals]
                Plotting.plot_summary_stats(ax,name,accepted_s=accepted_vals,s_obs=s_obs[i],s_hat=s_hat[i],dim=i)

    return fig

"""
    ABC-MCMC
"""
def abc_mcmc(n_obs:int,y_obs:[[float]],
    fitting_model:Models.Model,priors:["stats.Distribution"],
    chain_length:int,perturbance_kernels:"[function]",acceptance_kernel:"function",scaling_factor:float,
    summary_stats=None,distance_measure=l2_norm,show_plots=True,printing=True) -> (Models.Model,[[float]]):
    """
    DESCRIPTION
    Markov Chain Monte-Carlo Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    chain_length (int) - Length of markov chain to allow.
    perturbance_kernels ([function]) - Functions for varying parameters each monte-carlo steps.
    acceptance_kernel (function) - Function to determine whether to accept parameters
    scaling_factor (float) - Scaling factor for `acceptance_kernel`.

    OPTIONAL PARAMETERS
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)
    distance_measure - (func) - distance function to use (See choices above) (default=ABC.l2_norm)
    show_plots (bool) - whether to generate and show plots (default=True)
    printing (bool) - whether to print updates to terminal (default=True)

    RETURNS
    Model - fitted model with best parameters
    [[float]] - set of all accepted parameter values (use for further investigation)
    """
    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # find starting sample
    min_l1_norm=maxsize
    i=0
    while (True):
        if (printing): print("Finding Start - ({:,},{:,.3f})                       ".format(i,min_l1_norm),end="\r")
        i+=1
        theta_0=[pi_i.rvs(1)[0] for pi_i in priors]

        # observe theorised model
        fitting_model.update_params(theta_0)
        y_0=fitting_model.observe()
        s_0=[s(y_0) for s in summary_stats]

        # accept-reject
        norm_vals=[distance_measure(s_0_i,s_obs_i) for (s_0_i,s_obs_i) in zip(s_0,s_obs)]
        if (l1_norm(norm_vals)<min_l1_norm): min_l1_norm=l1_norm(norm_vals)
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factor)): break

    THETAS=[theta_0]
    ACCEPTED_SUMMARY_VALS=[s_0]

    if (printing): print("Found Start - ({:,})".format(i),theta_0)

    # MCMC step
    new=0
    for t in range(1,chain_length+1):
        # perturb last sample
        theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,THETAS[-1])]
        while any([p.pdf(theta)==0.0 for (p,theta) in zip(priors,theta_temp)]): theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,THETAS[-1])]

        # observed theorised model
        fitting_model.update_params(theta_temp)
        y_temp=fitting_model.observe()
        s_temp=[s(y_temp) for s in summary_stats]

        # accept-reject
        norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factor)):
            new+=1
            THETAS.append(theta_temp)
            ACCEPTED_SUMMARY_VALS.append(s_temp)
            if (printing): print("({:,}) - NEW".format(t),end="\r")
        else: # stick with last parameter sample
            THETAS.append(THETAS[-1])
            ACCEPTED_SUMMARY_VALS.append(ACCEPTED_SUMMARY_VALS[-1])
            if (printing): print("({:,}) - OLD".format(t),end="\r")

    theta_hat=list(np.mean(THETAS,axis=0))
    model_hat=fitting_model.copy(theta_hat)
    s_hat=[s(model_hat.observe()) for s in summary_stats]
    if (printing): print("{:.3f} observations were new.".format(new/chain_length))
    else: print("(New={:.3f})".format(new/chain_length),end="")

    if (show_plots):

        fig=plt.figure(constrained_layout=True)
        fig=__abc_mcmc_plotting(fig,y_obs,priors,fitting_model,model_hat,ACCEPTED_SUMMARY_VALS,THETAS)
        plt.show()

    return model_hat, THETAS

def __abc_mcmc_plotting(fig:plt.Figure,y_obs:[[float]],priors:["stats.Distribution"],fitting_model:Models.Model,model_hat:Models.Model,
                            accepted_summary_vals:[[float]],accepted_params:[[float]]) -> plt.Figure:

    n_simple_ss=sum(len(s)==1 for s in accepted_summary_vals[0]) # number of summary stats which map to a single dimension
    n_cols=3 if (n_simple_ss==0) else 4
    n_params=(fitting_model.n_params-2) if (type(fitting_model) is Models.SIRModel) else fitting_model.n_params
    n_rows=max([1,np.lcm.reduce([n_params,max(1,n_simple_ss),fitting_model.dim_obs])])

    gs=fig.add_gridspec(n_rows,n_cols)

    # plot fitted model
    row_step=n_rows//fitting_model.dim_obs
    for i in range(fitting_model.dim_obs):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        y_obs_dim=[y[i] for y in y_obs]
        Plotting.plot_accepted_observations(ax,fitting_model.x_obs,y_obs_dim,[],model_hat,dim=i)

    # plot traces & posteriors
    row_step=n_rows//n_params
    if (type(fitting_model) is Models.SIRModel):
        for i in range(2,fitting_model.n_params):
            name="Theta_{}".format(i)
            accepted_vals=[x[i] for x in accepted_params]
            # Chain trace
            ax=fig.add_subplot(gs[(i-2)*row_step:((i-2)+1)*row_step,0])
            Plotting.plot_MCMC_trace(ax,name,accepted_parameter=accepted_vals,predicted_val=model_hat.params[i])
            # Posteriors
            ax=fig.add_subplot(gs[(i-2)*row_step:((i-2)+1)*row_step,1])
            Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=model_hat.params[i],prior=priors[i],dim=i)
    else:
        for i in range(fitting_model.n_params):
            name="Theta_{}".format(i)
            accepted_vals=[x[i] for x in THETAS]
            # Chain trace
            ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
            Plotting.plot_MCMC_trace(ax,name,accepted_parameter=accepted_vals,predicted_val=model_hat.params[i])
            # Posteriors
            ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,1])
            Plotting.plot_parameter_posterior(ax,name,accepted_parameter=accepted_vals,predicted_val=model_hat.params[i],prior=priors[i],dim=i)

    # plot summary vals
    row=0
    row_step=n_rows//n_simple_ss
    for i in range(len(accepted_summary_vals[0])):
        if (len(accepted_summary_vals[0][i])==1):
            name="s_{}".format(i)
            ax=fig.add_subplot(gs[row*row_step:(row+1)*row_step,2])
            row+=1
            accepted_vals=[s[i][0] for s in accepted_summary_vals]
            Plotting.plot_summary_stats(ax,name,accepted_s=accepted_vals,s_obs=s_obs[i],s_hat=s_hat[i],dim=i)

    return fig

"""
    ABC-SMC
"""
def abc_smc(n_obs:int,y_obs:[[float]],
    fitting_model:Models.Model,priors:["stats.Distribution"],
    num_steps:int,sample_size:int,
    scaling_factors:[float],acceptance_kernel:"function",
    adaptive_perturbance=False,perturbance_kernels=None,perturbance_kernel_probability=None,
    summary_stats=None,distance_measure=l2_norm,show_plots=True,printing=True) -> (Models.Model,[[float]],[float]):
    """
    DESCRIPTION
    Sequential Monte-Carlo Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    num_steps (int) - Number of steps (ie number of scaling factors).
    sample_size (int) - Number of parameters samples to keep per step.
    scaling_factors ([float]) - Scaling factor for `acceptance_kernel`.
    acceptance_kernel (function) - Function to determine whether to accept parameters

    OPTIONAL PARAMETERS
    adaptive_perturbance (bool) - Whether to use an adaptive perturbance kernel. Overrules `perturbance_kernels` and `perturbance_kernel_probability` definitions (default=False)
    perturbance_kernels ([function]) - Functions for varying parameters each monte-carlo steps. (default=None (ie use adaptive))
    perturbance_kernel_probability ([function]) - Probability of x being pertubered to value y. (default=None (ie use adaptive))
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)
    distance_measure - (func) - distance function to use (See choices above)
    show_plots (bool) - whether to generate and show plots (default=True)
    printing (bool) - whether to print updates to terminal (default=True)

    RETURNS
    Model - fitted model with best parameters
    [[float]] - set of all accepted parameter values (use for further investigation)
    """
    # initial sampling
    if (num_steps!=len(scaling_factors)): raise ValueError("`num_steps` must equal `len(scaling_factors)`")
    if (not adaptive_perturbance) and (not perturbance_kernel_probability) and (not perturbance_kernels): raise ValueError("If `adaptive_perturbance` is `False` then you must define `perturbance_kernels` and `perturbance_kernel_probability`")

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]

    # initial sampling
    THETAS=[] # (weight,params)
    i=0
    while (len(THETAS)<sample_size):
        i+=1
        theta_temp=[pi_i.rvs(1)[0] for pi_i in priors]

        # observed theorised model
        fitting_model.update_params(theta_temp)
        y_temp=fitting_model.observe()
        s_temp=[s(y_temp) for s in summary_stats]

        # accept-reject
        norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factors[0])):
            THETAS.append((1/sample_size,theta_temp))
        if(printing): print("({:,}) - {:,}/{:,}".format(i,len(THETAS),sample_size),end="\r")
    if (printing): print()

    total_simulations=i

    # resampling & reweighting step
    for t in range(1,num_steps):
        i=0
        NEW_THETAS=[] # (weight,params)

        if (adaptive_perturbance): perturbance_kernels,perturbance_kernel_probability=__generate_smc_perturbance_kernels([x[1] for x in THETAS],printing)
        if (not printing): print("*",sep="",end="")

        while (len(NEW_THETAS)<sample_size):
            i+=1
            if (printing): print("({:,}/{:,} - {:,}) - {:,}/{:,} (eps={:.3f})".format(t,num_steps,i,len(NEW_THETAS),sample_size,scaling_factors[t]),end="\r")

            # sample from THETA
            new_i=np.random.choice([i for i in range(len(THETAS))],size=1,p=[weight for (weight,_) in THETAS])[0]
            theta_t=THETAS[new_i][1]

            # perturb sample
            theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,theta_t)]
            while any([p.pdf(theta)==0.0 for (p,theta) in zip(priors,theta_temp)]): theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,theta_t)]

            # observed theorised model
            fitting_model.update_params(theta_temp)
            y_temp=fitting_model.observe()
            s_temp=[s(y_temp) for s in summary_stats]

            # accept-reject
            norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
            if (acceptance_kernel(l1_norm(norm_vals),scaling_factors[t])):
                weight_numerator=sum([p.pdf(theta) for (p,theta) in zip(priors,theta_temp)])
                weight_denominator=0
                for (weight,theta) in THETAS:
                    weight_denominator+=sum([weight*p(theta_i,theta_temp_i) for (p,theta_i,theta_temp_i) in zip(perturbance_kernel_probability,theta,theta_temp)]) # probability theta_temp was sampled
                weight=weight_numerator/weight_denominator
                NEW_THETAS.append((weight,theta_temp))

        total_simulations+=i
        weight_sum=sum([w for (w,_) in NEW_THETAS])
        THETAS=[(w/weight_sum,theta) for (w,theta) in NEW_THETAS]
        if (printing): print()

    if (printing): print()

    param_values=[theta for (_,theta) in THETAS]
    weights=[w for (w,_) in THETAS]
    theta_hat=list(np.average(param_values,axis=0,weights=weights))
    model_hat=fitting_model.copy(theta_hat)

    if (printing):
        print("Total Simulations - {:,}".format(total_simulations))
        print("theta_hat -",theta_hat)

    if (show_plots):
        fig=plt.figure(constrained_layout=True)
        fig=__abc_smc_plotting(fig,y_obs,priors,fitting_model,model_hat,accepted_params=param_values,weights=weights)
        plt.show()

    return model_hat,param_values,weights

def __abc_smc_plotting(fig:plt.Figure,y_obs:[[float]],priors:["stats.Distribution"],fitting_model:Models.Model,model_hat:Models.Model,accepted_params:[[float]],weights:[float]) -> plt.Figure:

    n_params=(fitting_model.n_params-2) if (type(fitting_model) is Models.SIRModel) else fitting_model.n_params
    n_rows=max([1,np.lcm(n_params,fitting_model.dim_obs)])

    gs=fig.add_gridspec(n_rows,2)

    # plot fitted model
    row_step=n_rows//fitting_model.dim_obs
    for i in range(fitting_model.dim_obs):
        ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        y_obs_dim=[y[i] for y in y_obs]
        Plotting.plot_accepted_observations(ax,fitting_model.x_obs,y_obs_dim,[],model_hat,dim=i)


    row_step=n_rows//n_params
    if (type(fitting_model) is Models.SIRModel):
        for i in range(2,fitting_model.n_params):
            ax=fig.add_subplot(gs[(i-2)*row_step:(i-2+1)*row_step,0])
            name="theta_{}".format(i)
            accepted_parameter_values=[theta[i] for theta in accepted_params]
            Plotting.plot_parameter_posterior(ax,name,accepted_parameter_values,predicted_val=model_hat.params[i],prior=priors[i],dim=i,weights=weights)
    else:
        for i in range(fitting_model.n_params):
            ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
            name="theta_{}".format(i)
            parameter_values=[theta[i] for theta in accepted_params]
            Plotting.plot_parameter_posterior(ax,name,accepted_parameter_values,predicted_val=model_hat.params[i],prior=priors[i],dim=i,weights=weights)

    return fig

"""
    ADAPTIVE ABC-SMC
"""

def adaptive_abc_smc(n_obs:int,y_obs:[[float]],
    fitting_model:Models.Model,priors:["stats.Distribution"],
    max_steps:int,sample_size:int,acceptance_kernel:"function",alpha:float,
    initial_scaling_factor=maxsize,terminal_scaling_factor=0,max_simulations=None,
    summary_stats=None,distance_measure=l2_norm,show_plots=True,printing=True) -> (Models.Model,[[float]],float):
    """
    DESCRIPTION
    Fully adaptive Sequential Monte-Carlo Sampling version of Approximate Bayesian Computation for the generative models defined in `Models.py`.
    (Adaptive wrt perturbance kernel and bandwidths)

    PARAMETERS
    n_obs (int) - Number of observations available.
    y_obs ([[float]]) - Observations from true model.
    fitting_model (Model) - Model the algorithm will aim to fit to observations.
    priors (["stats.Distribution"]) - Priors for the value of parameters of `fitting_model`.
    max_steps (int) - Maximum number of resampling iterations (algorithm terminates if it reaches this value)
    sample_size (int) - Number of parameters samples to keep per step.
    acceptance_kernel (function) - Function to determine whether to accept parameters
    alpha (float) - Analogous to proportion of accepted samples to carry to next step. Used to determine acceptance kernel bandwidths. MUST be in (0,1).

    OPTIONAL PARAMETERS
    initial_scaling_factor (float in (0,1)) - Bandwidth the acceptance kernel begins at (default=maxsize)
    terminal_scaling_factor (float in (0,1)) - What value of bandwith to terminate the algorithm at. (default=0)
    max_simulations (int) - Maximum number of simulations (default=None=no limit). Only checked at the end of each iteration
    summary_stats ([function]) - functions which summarise `y_obs` and the observations of `fitting_model` in some way. (default=group by dimension)
    distance_measure - (func) - distance function to use (See choices above)
    show_plots (bool) - whether to generate and show plots (default=True)
    printing (bool) - whether to print updates to terminal (default=True)

    RETURNS
    Model - fitted model with best parameters
    [[float]] - set of all accepted parameter values (use for further investigation)
    """
    # initial sampling
    if (alpha<=0) and (alpha>=1): raise ValueError("`alpha` must be in (0,1)")

    group_dim = lambda ys,i: [y[i] for y in ys]
    summary_stats=summary_stats if (summary_stats) else ([(lambda ys:group_dim(ys,i)) for i in range(len(y_obs[0]))])
    s_obs=[s(y_obs) for s in summary_stats]
    scaling_factor=initial_scaling_factor

    # initial sampling
    THETAS=[] # (weight,params)
    distances=[]
    i=0
    while (len(THETAS)<sample_size):
        i+=1
        theta_temp=[pi_i.rvs(1)[0] for pi_i in priors]

        # observed theorised model
        fitting_model.update_params(theta_temp)
        y_temp=fitting_model.observe()
        s_temp=[s(y_temp) for s in summary_stats]

        # accept-reject
        norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
        if (acceptance_kernel(l1_norm(norm_vals),scaling_factor)):
            distances.append((1/sample_size,l1_norm(norm_vals)))
            THETAS.append((1/sample_size,theta_temp))
        if(printing): print("({:,}) - {:,}/{:,}".format(i,len(THETAS),sample_size),end="\r")
    if (printing): print()

    total_simulations=i

    # resampling & reweighting step
    t=0
    while(t<max_steps and scaling_factor>terminal_scaling_factor):
        if (not printing): print("*",sep="",end="")
        if (max_simulations and total_simulations>=max_simulations): break
        elif(printing): print("Total Sims = {:,} < {:,}\n".format(total_simulations,max_simulations))

        i=0
        NEW_THETAS=[] # (weight,params)

        perturbance_kernels,perturbance_kernel_probability=__generate_smc_perturbance_kernels([x[1] for x in THETAS],printing)
        scaling_factor=__calculate_scaling_factor(distances,acceptance_kernel,alpha)
        distances=[]

        while (len(NEW_THETAS)<sample_size):
            i+=1
            if (printing): print("({:,}/{:,} - {:,}) - {:,}/{:,} (eps={:,.3f}>{:,.3f})".format(t,max_steps,i,len(NEW_THETAS),sample_size,scaling_factor,terminal_scaling_factor),end="\r",flush=True)

            # sample from THETA
            new_i=np.random.choice([i for i in range(len(THETAS))],size=1,p=[weight for (weight,_) in THETAS])[0]
            theta_t=THETAS[new_i][1]

            # perturb sample
            theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,theta_t)]
            while any([p.pdf(theta)==0.0 for (p,theta) in zip(priors,theta_temp)]): theta_temp=[k(theta_i) for (k,theta_i) in zip(perturbance_kernels,theta_t)]

            # observed theorised model
            fitting_model.update_params(theta_temp)
            y_temp=fitting_model.observe()
            s_temp=[s(y_temp) for s in summary_stats]

            # accept-reject
            norm_vals=[distance_measure(s_temp_i,s_obs_i) for (s_temp_i,s_obs_i) in zip(s_temp,s_obs)]
            if (acceptance_kernel(l1_norm(norm_vals),scaling_factor)):
                weight_numerator=sum([p.pdf(theta) for (p,theta) in zip(priors,theta_temp)])
                weight_denominator=0
                for (weight,theta) in THETAS:
                    weight_denominator+=sum([weight*p(theta_i,theta_temp_i) for (p,theta_i,theta_temp_i) in zip(perturbance_kernel_probability,theta,theta_temp)]) # probability theta_temp was sampled
                weight=weight_numerator/weight_denominator
                NEW_THETAS.append((weight,theta_temp))
                distances.append((weight,l1_norm(norm_vals)))

        total_simulations+=i
        weight_sum=sum([w for (w,_) in NEW_THETAS])
        THETAS=[(w/weight_sum,theta) for (w,theta) in NEW_THETAS]
        distances=[(w/weight_sum,d) for (w,d) in distances]
        if (printing): print()
        t+=1

    if (printing): print()

    param_values=[theta for (_,theta) in THETAS]
    weights=[w for (w,_) in THETAS]
    theta_hat=list(np.average(param_values,axis=0,weights=weights))
    model_hat=fitting_model.copy(theta_hat)

    if (printing):
        print("Total Simulations - {:,}".format(total_simulations))
        print("theta_hat -",theta_hat)

    if (show_plots):
        fig=plt.figure(constrained_layout=True)
        fig=__abc_smc_plotting(fig,y_obs,priors,fitting_model,model_hat,accepted_params=param_values,weights=weights)
        plt.show()
        # n_rows=max([1,np.lcm(fitting_model.n_params,fitting_model.dim_obs)])
        #
        # fig=plt.figure(constrained_layout=True)
        # gs=fig.add_gridspec(n_rows,2)
        #
        # # plot fitted model
        # row_step=n_rows//fitting_model.dim_obs
        # for i in range(fitting_model.dim_obs):
        #     ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,-1])
        #     y_obs_dim=[y[i] for y in y_obs]
        #     Plotting.plot_accepted_observations(ax,fitting_model.x_obs,y_obs_dim,[],model_hat,dim=i)
        #
        #
        # row_step=n_rows//fitting_model.n_params
        # for i in range(fitting_model.n_params):
        #     ax=fig.add_subplot(gs[i*row_step:(i+1)*row_step,0])
        #     name="theta_{}".format(i)
        #     parameter_values=[theta[i] for theta in param_values]
        #     Plotting.plot_smc_posterior(ax,name,parameter_values=parameter_values,weights=weights,predicted_val=theta_hat[i],prior=priors[i],dim=i)
        #
        # plt.show()

    return model_hat,param_values,weights

def __generate_smc_perturbance_kernels(thetas:[[float]], printing=False) -> ([["function"]],[["function"]]):
    """
    DESCRIPTION
    Generate lambda functions to be used as perturbance kernels (and their associated pdfs) for ABC-SMC. These are the kernels recommened in Beaumont et al. 2009.
    Intended to be used for ABC-SMC with adaptive perturbance kernels.

    PARAMETERS
    thetas ([[float]]) - the set of parameters from previous sampling iteration.

    OPTIONAL PARAMTERS
    printing (bool) - whether to print log updates to console

    RETURNS
    [[func]],[[func]] - 1st=perturbance kernel functions, 2nd=pdfs for perturbance kernels
    """
    sigmas=np.var(thetas,ddof=1,axis=0)
    if (printing): print("Perturbance Variances=",sigmas,"                       ",sep="")
    perturbance_kernels=[lambda x:stats.norm(x,2*sigma).rvs(1)[0] for sigma in sigmas]
    perturbance_kernel_probability=[lambda x,y:stats.norm(0,2*sigma).pdf(x-y) for sigma in sigmas]

    return perturbance_kernels,perturbance_kernel_probability

def __calculate_scaling_factor(distances,acceptance_kernel,alpha) -> float:
    """
    DESCRIPTION
    Calculate a scaling factor for the next iteration of ABC-SMC given the set of distances which were accepted under the last scaling factor.
    I assume the acceptance kernel is symmetric and distances are positive.
    """
    if (acceptance_kernel is uniform_kernel):
        distances.sort(key=lambda x:x[1])
        u=0
        for (w,d) in distances:
            u+=w
            if (u>=alpha): return d
        return distances[-1][1]
    else:
        raise TypeError("`adaptive_abc_smc` is only properly implemented for the acceptance kernel to be `uniform_kernel`. Please choose this.")
