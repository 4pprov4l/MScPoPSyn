#import validationUtils
from Utils import plotUtils
from Utils import tuUtils
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def validate_Gan_Cat(sampled, real, train):
    categorical = ['MunicipalityOrigin']
    evaluate(real, sampled, categorical, train)
    categorical = ['ResphasDrivlic']
    evaluate(real, sampled, categorical, train)
    categorical = ['PopSocio']
    evaluate(real, sampled, categorical, train)
    categorical = ['MunicipalityOrigin', 'ResphasDrivlic']
    evaluate(real, sampled, categorical, train)
    categorical = ['MunicipalityOrigin', 'ResphasDrivlic', 'PopSocio']
    evaluate(real, sampled, categorical, train)
    categorical = ['MunicipalityOrigin', 'ResphasDrivlic', 'PopSocio', 'Gender']
    evaluate(real, sampled, categorical, train)
    categorical = ['HomeAdrZone']
    evaluate(real, sampled, categorical, train)
    categorical = ['Education']
    evaluate(real, sampled, categorical, train) 
    categorical = ['Sector']
    evaluate(real, sampled, categorical, train) 
    categorical = ['Education', 'Sector']
    evaluate(real, sampled, categorical, train) 
    categorical = ['Education', 'Sector', 'Gender']
    evaluate(real, sampled, categorical, train) 

def validate_Gan_num(real, sampled):
    numerical = ['WorkHoursPw', 
             'FamNumAdults', 'FamNumPers', 'HousehNumAdults', 'HousehNumcars', 
             'HousehNumPers', 'RespAgeCorrect']
    
    incVar = ['IncFamily2000', 'IncRespondent2000']
    
    for var in incVar:
        print('Variable: ' + var)
        print(stats.ttest_ind(sampled[var], real[var]))
        plt.hist([sampled[var].loc[~is_outlier(sampled[var])], real[var].loc[~is_outlier(real[var])]],label=['Sampled', 'test'], density=True)
        plt.legend(loc='upper right')
        plt.show()

    for var in numerical:
        print('Variable: ' + var)
        print(stats.ttest_ind(sampled[var], real[var]))
        plt.hist([sampled[var], real[var]], 50,label=['Sampled', 'test'], density=True)
        plt.legend(loc='upper right')
        plt.show()


def structuralAndSamplingZeros():
    return 0

def evaluate(real_df, model_df, agg_vars, train):

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
    print('Aggregated variables: ')
    for var in agg_vars:
        N *= train[var].nunique() 
        print(var)
    print('Number of combinations: ' + str(N))

    #metrics = {}
    #metrics['MAE'] = np.sum(abs(diff))/N
    #metrics['MSE'] = np.sum(diff**2)/N
    #metrics['RMSE'] = np.sqrt(metrics['MSE'])
    #metrics['SRMSE'] = np.sqrt(metrics['MSE']*N)
    #print('MAE:{}, MSE:{}, RMSE:{}, SRSME:{}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['SRMSE']))
    
    stats = plotUtils.compute_stat(real_and_sampled['count_real'], real_and_sampled['count_sampled'],do_plot=True, plot_log=False)#, plot_name='_'.join(['VAE']+agg_vars))
        
    return real_and_sampled, stats

def is_outlier(points, thresh=8):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    median = np.median(points)
    diff = (points - median)**2
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def validateModel(real, samples, numerical):
    concatenated_df = pd.concat([real, samples])

    samples = tuUtils.numerical_binning(concatenated_df, numerical, q=20)

    split_index = len(real)
    real = concatenated_df.iloc[:split_index, :]
    samples = concatenated_df.iloc[split_index:, :]

    stats = pd.DataFrame()


    for col in real.columns:
        new = evaluateModel(real, samples, col)
        new['col'] = col
        new = pd.DataFrame([new])
        stats = pd.concat([stats, new]) 

    stats.set_index('col', inplace=True)
                          
    return stats

def evaluateModel(real_df, model_df, agg_vars):

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
    
    stats = plotUtils.compute_stat(real_and_sampled['count_real'], real_and_sampled['count_sampled'],do_plot=False, plot_log=False)#, plot_name='_'.join(['VAE']+agg_vars))
        
    return  stats












