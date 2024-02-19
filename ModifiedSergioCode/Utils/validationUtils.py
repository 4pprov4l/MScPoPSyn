import numpy as np
import pandas as pd
import plotUtils
import TUutils
from TUutils import samples_to_df

# This script implements functions to perform some form of validation.
## Part 1 of this script are the "benchmark" models which are a fully random sampler, margin sampler and gibbs sampler
## Part 2 of this script implements an "evaluator" function which computes MAE, MSE RMSE and SRMSE for the joint distributions of the sampled set and the test set and plots these distributions
## Part 3 of this script includes a duplicate checker between the model and some other dataset with the same structure

def random_sampler(n_samples, pre_one_hot_df): # Random sampling for benchmark
    pre_one_hot_df = pre_one_hot_df.astype(str)
    samples = {}
    for var in list(pre_one_hot_df): # Looping over all variables
        categories =  list(pre_one_hot_df[var].unique()) # Getting the categories from the variables
        samples['{}'.format(var)] = np.random.choice(a=categories, size=n_samples) # Replace the row with the new value
    samples = pd.DataFrame.from_dict(samples)
    samples = samples[list(samples)].astype('category')
    
    return samples


def margin_sampler(n_samples, pre_one_hot_df): # Margin sampling for benchmark
    samples = {}
    margins_df = pre_one_hot_df.astype(str).copy()
    margins_df['count'] = 1
    for var in list(pre_one_hot_df): # Looping over all variables
        #marginals = margins_df[[var, 'count']]
        #marginals = marginals.groupby(var, observed=True).count()/pre_one_hot_df.shape[0]
        #categories =  list(marginals.reset_index()[var]) # Getting the categories from the variables
        #samples['{}'.format(var)] = np.random.choice(a=categories, size=n_samples, p=marginals['count']) # Replace the row with the new value
        samples['{}'.format(var)] = np.random.choice(a=margins_df[var], size=n_samples) # Replace the row with the new value
    samples = pd.DataFrame.from_dict(samples)
    samples = samples[list(samples)].astype('category')

    return samples


def gibbs_sampler(n_samples, agg_vars, pre_one_hot_df, n_restarts=1):
    
    pre_one_hot_df['counts'] = 1
    pre_one_hot_df = pre_one_hot_df[agg_vars + ['counts']].groupby(agg_vars, observed=True).count()
    pre_one_hot_df.reset_index(inplace=True)
    gibbs = pre_one_hot_df[agg_vars].sample() # Sample initialization and restarting 

    #### Gibbs sampler
    for n in range(1, n_samples):
        if n in (np.linspace(0, n_samples, n_restarts).astype('int')):
            gibbs = gibbs.append(pre_one_hot_df[agg_vars].sample()) # Sample initialization
            continue
        gibbs = gibbs.append(gibbs.iloc[[n-1]]).reset_index(drop=True)
        for s_v in agg_vars:
            temp_vars = list(set(agg_vars) - set([s_v])) # List of fixed variables  
            i1 = pre_one_hot_df.set_index(temp_vars).index # Make the index of the complete dataset the temporal vars
            i2 = gibbs.set_index(temp_vars).index    # Make the index of the gibbs samples the temporal vars

            dist = (pre_one_hot_df['counts'][i1.isin(i2)]/pre_one_hot_df['counts'][i1.isin(i2)].sum()) # Get the distribution on the counts
            if all(dist==0.): # Just in case all of them are 0's
                idx = np.random.choice(dist.index)
            else: 
                idx = np.random.choice(dist.index, p=dist) #
            gibbs.loc[n, s_v] = pre_one_hot_df[agg_vars].loc[idx, s_v] # Replace the row with the new value
    
    return gibbs


def evaluate(real, model, agg_vars, col_names, original_df, pre_one_hot_df, n_samples=None, plot=True): # Evaluate the real data and the model data using certain evaluation metrics and plot
    ##### Returning to original format
    try: # If model is a class
        if n_samples is not None:
            model.n_samples = n_samples
        samples = model.sampler()
        model_df = TUutils.samples_to_df(samples, col_names=col_names, original_df=original_df, pre_one_hot_df=pre_one_hot_df)
    except: # If model are samples (such as the train set or marginal sampler set)
        try:
            model_df = TUutils.samples_to_df(model.transpose(), col_names=col_names, original_df=original_df, pre_one_hot_df=pre_one_hot_df)
        except: # If the samples are already in the form of the dataset
            model_df = model
    
    real_df = TUutils.samples_to_df(real.transpose(), col_names=col_names, original_df=original_df, pre_one_hot_df=pre_one_hot_df)

    ##### Adding a counter variable and aggregating to retrieve probabilities
    model_df['count'] = 1
    model_df = model_df.groupby(agg_vars, observed=True).count()
    model_df /= model_df['count'].sum()

    real_df['count'] = 1
    real_df = real_df.groupby(agg_vars, observed=True).count()
    real_df /= real_df['count'].sum()

    ##### Merge and difference
    real_and_sampled = pd.merge(real_df, model_df, suffixes=['_real', '_sampled'], on=agg_vars, how='outer', indicator=True)
    real_and_sampled = real_and_sampled[['count_real', 'count_sampled']].fillna(0)
    real_and_sampled['diff'] = real_and_sampled.count_real-real_and_sampled.count_sampled
    diff = np.array(real_and_sampled['diff'])
    
    N = 1
    for var in agg_vars:
        N *= len(pre_one_hot_df[var].cat.categories) 

    metrics = {}
    metrics['MAE'] = np.sum(abs(diff))/N
    metrics['MSE'] = np.sum(diff**2)/N
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['SRMSE'] = np.sqrt(metrics['MSE']*N)
    print('MAE:{}, MSE:{}, RMSE:{}, SRSME:{}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['SRMSE']))
    
    if plot:
        plotUtils.compute_stat(real_and_sampled['count_real'], real_and_sampled['count_sampled'], do_plot=True, plot_log=False)#, plot_name='_'.join(['VAE']+agg_vars))
        
    return real_and_sampled


def check_individuals(train, model, pre_one_hot_df, original_df, col_names):
    train_df = samples_to_df(train.transpose(), print_duplicates=False)
    try:
        samples  = model.sampler()
        model_df = TUutils.samples_to_df(samples.transpose(), col_names=col_names, original_df=df, pre_one_hot_df=pre_one_hot_df)
    except:
        model_df = TUutils.samples_to_df(model.transpose(), col_names=col_names, original_df=df, pre_one_hot_df=pre_one_hot_df)
    
    train_model = pd.merge(train_df, model_df, on=list(pre_one_hot_df), how='outer', indicator=True)
    print(train_model.head())
    print(train_model.shape)

    print("The numer of values on the train set NOT present on the model set is: {}".format(sum(real_model['_merge']=='left_only')))
    print("The number of values on the model set NOT present on the test set are: {}".format(sum(real_model['_merge']=='right_only')))
    print(sum(train_model['_merge']=='both'))