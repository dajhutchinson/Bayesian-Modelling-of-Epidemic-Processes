import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_accepted_observations(ax:plt.Axes,x_obs:int,y_obs:[[float]],accepted_observations:[[float]],predicted_model:"Model",dim=0) -> plt.Axes:
    """
    DESCRIPTION
    plot observations from truth `y_obs` and observations from accepted parameter sets.

    PARAMETERS
    ax (plt.Axes) - axes to plot on.
    x_obs (int) -
    y_obs ([float]) - observations from true model.
    accepted_observations ([[float]]) - observations from accepted parameter sets.
    dim (int) - dimension of observations being plotted (default=0)

    RETURNS
    plt.Axes - axes on which plot was made
    """
    if (accepted_observations!=[]): ax.scatter([],[],c="blue",alpha=1,marker="x",label="Accepted")
    for obs in accepted_observations:
        ax.scatter(x_obs,obs,c="blue",alpha=.05,marker="x")
    ax.scatter(x_obs,y_obs,c="green",alpha=1,label="y_obs")

    y_pred=predicted_model.observe(inc_noise=False)
    if (len(y_pred[0])!=1): # multi-dimensional
        y_pred=[y[dim] for y in y_pred]
    ax.plot(x_obs,y_pred,c="orange",label="Prediction")

    ax.set_title("Accepted Observations (dim={})".format(dim))
    ax.set_xticks([])
    ax.set_xticklabels([])
    if (dim==0): ax.legend()
    ax.margins(0)

    return ax

def plot_parameter_posterior(ax:plt.Axes,name:str,accepted_parameter:[float],predicted_val:float,prior:"stats.Distribution",dim=0) -> plt.Axes:
    """
    DESCRIPTION
    plot posterior of a parameter.

    PARAMETERS
    ax (plt.Axes) - axes to plot on.
    name (str) - name of parameter.
    accepted_parameter ([float]) - values of parameter which were accepted during sampling.
    predicted_val (float) - predicted value for parameter (likely mean of `accepted_parameter`)
    prior (stats.Distribution) - prior used when sampling for parameter.

    RETURNS
    plt.Axes - axes on which plot was made
    """
    # plot prior used
    x=np.linspace(min(accepted_parameter+[prior.ppf(.01)-1]),max(accepted_parameter+[prior.ppf(.99)+1]),100)
    # x=np.linspace(prior.ppf(.01),prior.ppf(.99),100)
    ax.plot(x,prior.pdf(x),"k-",lw=2, label='Prior')

    # plot accepted  points
    ax.hist(accepted_parameter,density=True)

    # plot smooth posterior (ie KDE)
    density=stats.kde.gaussian_kde(accepted_parameter)
    ax.plot(x,density(x),"--",lw=2,c="orange",label="Posterior KDE")

    ymax=ax.get_ylim()[1]
    ax.vlines(predicted_val,ymin=0,ymax=ymax,colors="orange",label="Prediction")
    ax.set_xlabel(name)
    ax.set_title("Posterior for {}".format(name))
    if (dim==0): ax.legend()
    ax.margins(0)

    return ax

def plot_summary_stats(ax:plt.Axes,name:str,accepted_s:[float],s_obs:float,s_hat:float,dim=0) -> plt.Axes:
    """
    DESCRIPTION
    plot values of a summary statistic generated by accepted parameter values during sampling.

    PARAMETERS
    ax (plt.Axes) - axes to plot on.
    name (str) - name of summary statistic.
    accepted_s ([float]) - values of summary statistic from sampling
    s_obs (float) - summary statistic values from true model.
    prior (stats.Distribution) - summary statistic value of fitted model.

    RETURNS
    plt.Axes - axes on which plot was made
    """
    ax.hist(accepted_s)
    ymax=ax.get_ylim()[1]
    ax.vlines(s_obs,ymin=0,ymax=ymax,colors="green",label="s_obs")
    ax.vlines(s_hat,ymin=0,ymax=ymax,colors="orange",label="From Fitted")

    ax.set_xlabel(name)
    ax.set_title("Accepted {}".format(name))
    if (dim==0): ax.legend()
    ax.margins(0)

    return ax

def plot_MCMC_trace(ax:plt.Axes,name:str,accepted_parameter:[float],predicted_val:float) -> plt.Axes:
    """
    DESCRIPTION
    plot the parameter value used in each step of MCMC process.

    PARAMETERS
    ax (plt.Axes) - axes to plot on.
    name (str) - name of parameter.
    accepted_parameter ([float]) - values of parameter which were accepted during sampling.
    predicted_val (float) - predicted value for parameter (likely mean of `accepted_parameter`)

    RETURNS
    plt.Axes - axes on which plot was made
    """
    x=list(range(1,len(accepted_parameter)+1))
    ax.plot(x,accepted_parameter,c="black")
    ax.hlines(predicted_val,xmin=0,xmax=len(accepted_parameter),colors="orange")

    ax.set_ylabel(name)
    ax.set_xlabel("t")
    ax.set_title("Trace {}".format(name))
    ax.margins(0)

    return ax

def plot_smc_posterior(ax:plt.Axes,name:str,parameter_values:[float],weights:[float],predicted_val:float,prior:"stats.Distribution",dim=0) -> plt.Axes:
    """
    DESCRIPTION
    plot posterior of a parameter from SMC algorithm.

    PARAMETERS
    ax (plt.Axes) - axes to plot on.
    name (str) - name of parameter.
    parameter_values ([float]) - values of parameter used.
    weights ([float]) - weights of each parameter used (must align with `parameter_values`)
    predicted_val (float) - predicted value for parameter (likely mean of `parameter_values` weighted by `weights`)
    prior (stats.Distribution) - prior used when sampling for parameter.

    RETURNS
    plt.Axes - axes on which plot was made
    """
    # plot prior used
    x=np.linspace(prior.ppf(.01)-1,prior.ppf(.99)+1,100)
    # x=np.linspace(prior.ppf(.01),prior.ppf(.99),100)
    ax.plot(x,prior.pdf(x),"k-",lw=2, label='Prior')

    density=stats.kde.gaussian_kde(parameter_values,weights=weights)
    x=np.linspace(min(parameter_values)*.9,max(parameter_values)*1.1,100)
    ax.plot(x,density(x),c="blue",label="Posterior")

    ymax=ax.get_ylim()[1]
    ax.vlines(predicted_val,ymin=0,ymax=ymax,colors="orange",label="Predicted Value")
    ax.set_xlabel(name)
    ax.set_ylabel("P")
    ax.set_title("Posterior for {}".format(name))
    ax.margins(0)
    if (dim==0): ax.legend()

    return ax

def plot_sir_model(ax:plt.Axes, model:"SIRModel",include_susceptible=True):

    ax.margins(0)

    xs=model.x_obs
    ys=model.observe()

    y_min=0,
    if (include_susceptible): y_max=np.ceil(model.population_size/1000)*1000
    else:
        y_max=max([max(y[1:]) for y in ys])
        mag=np.floor(np.log10(y_max))
        y_max=(10**mag)*np.ceil(y_max/(10**mag))

    x_min=min(xs,key=lambda x:x[0])[0]
    x_max=max(xs,key=lambda x:x[0])[0]

    # S
    if (include_susceptible):
        i=0
        y_obs=[y[i] for y in ys]
        ax.scatter(xs,y_obs,c="green",label="Susceptible")

    # I
    i=1
    y_obs=[y[i] for y in ys]
    ax.scatter(xs,y_obs,c="blue",label="Infectious")

    # R
    i=2
    y_obs=[y[i] for y in ys]
    ax.scatter(xs,y_obs,c="red",label="Removed")

    ax.set_title("Realisation of Population Sizes for Standard SIR Model",fontsize=20)

    ax.set_xlabel("Time-Period",fontsize=16)
    ax.set_ylabel("Population Size",fontsize=16)

    ax.set_xticks(list(range(x_min,x_max,7))+[x_max])
    ax.set_yticks(np.linspace(0, y_max, 5))
    ax.set_yticklabels(["{:,.0f}".format(x) for x in np.linspace(0, y_max, 5)])

    ax.legend()
    ax.grid()

    return

def stochastic_sir_model_realisations(ax,model,n_reals,labels=False):
    xs=model.x_obs

    y_min=0,
    y_max=np.ceil(model.population_size/1000)*1000

    x_min=min(xs,key=lambda x:x[0])[0]
    x_max=max(xs,key=lambda x:x[0])[0]

    a=.2

    models=[model.copy(list(model.params)) for _ in range(0,n_reals)]

    for m in models:
        ys=m.observe()

        # S
        i=0
        y_obs=[y[i] for y in ys]
        if labels: ax.plot(xs,y_obs,c="green",label="Susceptible",alpha=a)
        else: ax.plot(xs,y_obs,c="green",alpha=a)

        # I
        i=1
        y_obs=[y[i] for y in ys]
        if labels: ax.plot(xs,y_obs,c="blue",label="Infectious",alpha=a)
        else: ax.plot(xs,y_obs,c="blue",alpha=a)

        # R
        i=2
        y_obs=[y[i] for y in ys]
        if labels: ax.plot(xs,y_obs,c="red",label="Removed",alpha=a)
        else: ax.plot(xs,y_obs,c="red",alpha=a)

    f = lambda m,l,c: ax.plot([],[], marker=m,color=c,ls=l)[0]
    handles = [f("s","",c) for c in ["green","blue","red"]]

    ax.legend(handles=handles,labels=["S","I","R"],loc="upper left",ncol=1)

    ax.margins(0)
    ax.set_title("{} Realisations of Population Sizes for Standard SIR Model".format(n_reals),fontsize=20)

    ax.set_xlabel("Time-Period",fontsize=16)
    ax.set_ylabel("Population Size (,000s)",fontsize=16)

    ax.set_xticks(list(range(x_min,x_max,7))+[x_max])
    ax.set_yticks(np.linspace(0, y_max, 5))
    ax.set_yticklabels(["{:,.0f}".format(x) for x in np.linspace(0, y_max, 5)])

    return ax