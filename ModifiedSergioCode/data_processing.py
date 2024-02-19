### Basic module import
import os           # Working directory
import pandas as pd # Data processing
import numpy as np  # Scientific computing/matrix algebra
import matplotlib
import matplotlib.pyplot as plt # Common graphing interface (check also plotly and plotnine)
import re # Regular expressions

### Changing the working directory
#os.chdir('C:/Users/shgm/Desktop/projects/job_VAE') # PC
#os.chdir('/mnt/sdb1/data_shgm/') # Server 32
#os.chdir('/mnt/sdb1/data_sergio/') # Server ai
#print(os.getcwd())

### Reading the job data
job_df = pd.read_csv('job.csv', sep=',')

# Defining the 'categorical' and 'numerical' lists and changing the datatypes accordingly
categorical = ['Municipality', 'AgeGroup', 'Gender', 'Sector', 'Socio']
numerical   = ['Val', 'Year']

job_df[categorical] = job_df[categorical].astype('category')

# Saving this set for further investigation
job_mun = set(job_df['Municipality'])

# Read the population data
pop_df = pd.read_csv('pop.csv', sep=',')

# Defining the 'categorical' and 'numerical' lists and changing the datatypes accordingly
categorical = ['Municipality', 'edu', 'PopSocio', 'Sector', 'AgeGroup', 'Gender']

pop_df[categorical] = pop_df[categorical].astype('category')

pop_mun = set(pop_df['Municipality'])

# Reading of the commuting matrix
#del job_mun, pop_mun
cm_df = pd.read_csv('commuter_codes.csv', sep=',')

# Defining the 'categorical' and 'numerical' lists and changing the datatypes accordingly
categorical = ['Gender', 'Residence', 'Work']
numerical = ['Year', 'value']

cm_df[categorical] = cm_df[categorical].astype('category')

# Commuter matrix with names instead of numbers
cm_df_val = pd.read_csv('commuter_values.csv', sep=',', encoding = "ISO-8859-1")

# Create a serial number to merge
cm_df['id'] = cm_df.index
cm_df_val['id'] = cm_df_val.index

# Merge the databases
cm_df_tot = cm_df.merge(cm_df_val, on='id', suffixes=('_c', '_v'))

# Convert the residence code for further work
cm_df_tot['Residence_c'] = cm_df_tot['Residence_c'].astype('int64')

names_codes = cm_df_tot[['Work_c', 'Work_v']].drop_duplicates()

# TU
ses_df = pd.read_csv('session.csv', sep=',') # 140672 obs
# ses_df = pd.read_csv('session.txt', sep=';', encoding = "ISO-8859-1")

# Defining the 'categorical' and 'numerical' lists and changing the datatypes accordingly.

# PC
# categorical = ['RespSex', 'RespEdulevel', 'RespPrimOcc', 'HomeAdrMunCode', 'PrimOccMuncode', 'HousehAccomodation']

# Server
categorical = ['RespSex', 'RespEdulevel', 'RespPrimOcc', 'HomeAdrMunCode', 'PrimOccMuncode']
numerical = ['RespAgeCorrect', 'IncRespondent', 'SessionWeight', 'DiaryYear', 'TotalMin']

all_vars = categorical + numerical

#ses_df = ses_df[all_vars]
ses_df[categorical] = ses_df[categorical].astype('category')

# What are the values on home and occupation codes on the TU data
occ_cat = set(list(ses_df.PrimOccMuncode.cat.categories))
hom_cat = set(list(ses_df.HomeAdrMunCode.cat.categories))

# Which of these appear on the names_codes dataframe we created earlier?
diff = list(set(names_codes.Work_c) ^ occ_cat)
diff = [int(i) for i in diff]

names_codes.loc[names_codes['Work_c'].isin(diff)]

## At first, it seems that we dont need any of the pop and cm matrices we have. We could just aggregate the data on the TU
#   dataset and run the gibbs sampling on it. This needs further discussion.

ses_df = ses_df.drop(ses_df[ses_df.PrimOccMuncode.isin(diff) | # Occupation municipality
                            ses_df.HomeAdrMunCode.isin(diff)].index) # Home municipality

ses_df[categorical].describe()._append(ses_df[categorical].isnull().sum().rename('isnull'))

# Na's from the session database
# Any variable
for var in categorical:
    ses_df.drop(ses_df[ses_df[var].isnull()].index, inplace=True)
# Uknown category from occupation
ses_df.drop(ses_df[ses_df['RespPrimOcc']==9.].index, inplace=True)   

# Na's from the population database
pop_df.drop(pop_df[pop_df.edu=='H90'].index, inplace=True) # Non-stated education
pop_df.drop(pop_df[pop_df.Sector=='X'].index, inplace=True) # Non-stated activity
pop_df['Sector'] = pop_df.Sector.cat.remove_unused_categories() # Remove unused categories

# Recoding of categorical variables
#### Job
job_df['AgeGroup'] = np.where(job_df['AgeGroup']=='-15', '15-19', job_df['AgeGroup'])

# Turning it back again to category
job_df['AgeGroup'] = job_df['AgeGroup'].astype('category')

#### Pop
# Education
pop_df['edu'] = pop_df['edu'].astype(str)
pop_df['edu'][(pop_df['edu'] =='H70') | (pop_df['edu'] =='H80')] = 'H99'
pop_df['edu'] = pop_df['edu'].astype('category')

#### TU
# Age
age_bins = [-1, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 200]
age_cats = list(pop_df.AgeGroup.cat.categories)

# Create the categorical variable
ses_df['AgeGroup'] = pd.cut(ses_df.RespAgeCorrect, age_bins, labels=age_cats)

# Education, still in process
ses_df['edu'] = ''
ses_df['edu'][(ses_df['RespEdulevel'] ==1.) | 
                    (ses_df['RespEdulevel']==2.) | 
                    (ses_df['RespEdulevel']==3.) | 
                    (ses_df['RespEdulevel']==4.)] = 'H10'
ses_df['edu'][ses_df['RespEdulevel'] ==5.] = 'H20'
ses_df['edu'][ses_df['RespEdulevel'] ==6.] = 'H50'
ses_df['edu'][ses_df['RespEdulevel'] ==9.] = 'H35'
ses_df['edu'][ses_df['RespEdulevel'] ==11.] = 'H30'
ses_df['edu'][ses_df['RespEdulevel'] ==12.] = 'H40'
ses_df['edu'][(ses_df['RespEdulevel'] ==13.) | (ses_df['RespEdulevel'] ==14.)] = 'H99'

# Reset it to be a category
ses_df['edu'] = ses_df['edu'].astype('category')

# Main occupation of the respondent
ses_df['PopSocio'] = ''
ses_df['PopSocio'][(ses_df['RespPrimOcc'] ==1.) | 
                    (ses_df['RespPrimOcc']==2.) | 
                    (ses_df['RespPrimOcc']==3.)] = '0'
ses_df['PopSocio'][(ses_df['RespPrimOcc'] ==22.) | 
                    (ses_df['RespPrimOcc']==30.) | # ?
                    (ses_df['RespPrimOcc']==50.) | # ?
                    (ses_df['RespPrimOcc']==52.)] = '1' # ?
ses_df['PopSocio'][(ses_df['RespPrimOcc'] ==11.)] = '2' # ?
ses_df['PopSocio'][(ses_df['RespPrimOcc'] ==15.) | 
                    (ses_df['RespPrimOcc']==20.) | # ?
                    (ses_df['RespPrimOcc']==12.) | # ?
                    (ses_df['RespPrimOcc']==10.)] = '3' # ?

# Reset it to be a category
ses_df['PopSocio'] = ses_df['PopSocio'].astype('category')

# RespSex, change the names of the categories
ses_df['RespSex'] = ses_df.RespSex.cat.rename_categories({1.: 'M', 2.: 'K'})

# Data aggregation
# Pop database
pop_agg = ['MunicipalityOrigin', 'edu', 'PopSocio', 'AgeGroup', 'Gender', 'Year'] # Variables we group by

pop_df = pop_df.rename(index=str, columns={'Municipality':'MunicipalityOrigin'})
pop_df = pop_df.groupby(pop_agg + ['Sector'], as_index=False).sum()

# Change the names of the sesssion database in order to make an easier merge
ses_names = {
    'PrimOccMuncode':'MunicipalityDest',
    'HomeAdrMunCode':'MunicipalityOrigin',
    'RespSex':'Gender',
    'DiaryYear':'Year',
}
ses_df = ses_df.rename(index=str, columns=ses_names)

# Merge:
# We get the combinations totals
pop_df['sums'] = pop_df.groupby(pop_agg, as_index=False)['Val'].transform(sum)

# Replace the zero valued sums with 1s
pop_df['sums'][(pop_df['sums']==0.)] = 1

# We divide the values by the totals to get the percentages
pop_df['percent'] = pop_df.Val/pop_df.sums
pop_df = pop_df.drop(['Val', 'sums'], axis=1)

# Make copies of the databases and make the indices of the databases the common variables.
idx_list = ['MunicipalityOrigin', 'edu', 'PopSocio', 'AgeGroup', 'Gender', 'Year']

# We convert the data types of the variables that don't have the same type, otherwise, merging won't be possible.
for var in ['MunicipalityOrigin', 'edu', 'PopSocio', 'AgeGroup', 'Gender']: 
    pop_df[var] = pop_df[var].astype(str)
    pop_df[var] = pop_df[var].astype('category')
    ses_df[var] = ses_df[var].astype(str)
    ses_df[var] = ses_df[var].astype('category')
    
ses_df_c = ses_df[ses_df.Year>2008].set_index(idx_list)
pop_df_c = pop_df.copy().set_index(idx_list)

print(len(ses_df_c))
print(len(pop_df_c))

# Merge the databases so we have a distribution of industries for each person on TU
merged = ses_df_c.join(pop_df_c, how='inner')
print(len(merged))
# Drop unnecesary variables
merged = merged[['SessionId', 'Sector', 'percent']].reset_index(drop=True)

# We define the sampling function. This function is necessary when we want to aggregate the data by index
def groupby_sample(x):
    if all(x.percent==0.): # Just in case all of them are 0's
        return np.random.choice(x.Sector)
    else: 
        return np.random.choice(x.Sector, p=x.percent)

# This is the final dataframe. There are as many observations as the original TU data, but in this particular database, we have industries for each person too.
samp_df = merged.groupby('SessionId', as_index=False).aggregate(groupby_sample)
samp_df = samp_df.drop(['percent'], axis=1)

samp_df = samp_df.merge(ses_df_c.reset_index(), on='SessionId', how='inner')

# First dataset
samp_df.to_csv('sampling_df.txt', sep=',', index=False)
samp_df = pd.read_csv('sampling_df.txt', sep=',')


# After analayzing the correlation between variables, variables with more than 95% of correlation were dropped from the database. This, hopefully will reduce the dimensionality of the input

#drop = ['JstartMuncode', 'JstartNTMzone', 'HomeAdrNTMzone', 'SduMuncode', 'SduOldMuncode', 'HomeAdrOldMuncode', 'DayStartOldMuncode', 'SduNTMzone', 'PrimOccNTMzone', 
#'PrimOccOldMuncode', 'IncNuclFamily2000', 'IncFamily2000', 'IncFamily', 'DayStartMuncode', 'DayStartNTMzone', 'IncHouseh2000', 'IncNuclFamily', 'IncHouseh', 
#'DayPrimTargetMuncode', 'DiaryDate', 'FamNumPers', 'HousehNumDrivLic', 'HousehNumAdults', 'FamNumPers1084', 'RespAgeSimple', 'HousehNumPers', 'IncRespondent', 
#'IncSpouse2000', 'RespYearBorn', 'NuclFamNumPers', 'IncSpouse', 'TotalLen', 'NuclFamNumPers1084', 'RespPrimOcc', 'RespEdulevel'] 

drop = ['JstartMuncode', 'SduMuncode', 'IncNuclFamily2000', 'IncFamily2000', 'IncFamily', 'DayStartMuncode', 'IncHouseh2000', 'IncNuclFamily', 'IncHouseh', 
'DayPrimTargetMuncode', 'DiaryDate', 'FamNumPers', 'HousehNumDrivLic', 'HousehNumAdults', 'FamNumPers1084', 'RespAgeSimple', 'HousehNumPers', 'IncRespondent', 
'IncSpouse2000', 'RespYearBorn', 'NuclFamNumPers', 'IncSpouse', 'TotalLen', 'NuclFamNumPers1084', 'RespPrimOcc', 'RespEdulevel'] 

samp_df.drop(drop, inplace=True, axis=1)

### Numerical variables define
numerical = ['IncRespondent2000', 'TotalLenExclComTrans', 'TotalMotorLen', 'TotalBicLen', 'TotalMin', 'TotalMotorMin', 'SessionId', 'WorkHoursPw', 'GISdistHW', 'HousehNumPers1084',
            'TotalNumTrips', 'NumTripsCorr', 'NumTripsExclComTrans', 'SessionWeight', 'HomeAdrCitySize', 'HomeAdrDistNearestStation', 'HwDayspW', 'WorkatHomeDayspM', 
             'JstartDistNearestStation', 'NightsAway'] # Numerical variables with missing values 
samp_df[numerical] = samp_df[numerical].astype('float64')

def replace_missing(df, grouping_vars = ['MunicipalityOrigin', 'AgeGroup', 'edu', 'PopSocio']):
    """
    Replaces the missing values of numerical variables on the dataframe by taking the mean (or could be median) of the variables grouped by certain variables.
    It makes a two dataframes: one with the nans which we use to replace the missing values and one with the rest of the variables. 
    """
    # Variables we groupby 
    numerical = ['IncRespondent2000', 'TotalLenExclComTrans', 'TotalMotorLen', 'TotalBicLen', 'TotalMin', 'TotalMotorMin', 'WorkHoursPw', 'GISdistHW', 'HousehNumPers1084',
            'TotalNumTrips', 'NumTripsCorr', 'NumTripsExclComTrans', 'SessionWeight', 'HomeAdrCitySize', 'HomeAdrDistNearestStation', 'HwDayspW', 'WorkatHomeDayspM', 
             'JstartDistNearestStation', 'NightsAway'] # Numerical variables with missing values 
    
    # Divide the dataset
    df_auxiliar = df.drop(numerical, axis=1) # Dataframe which we don't modify
    df_missing  = df[grouping_vars+numerical+['SessionId']] # Dataframe with missings

    # NaN replacement
    f = lambda x: x.fillna(np.random.choice(x)) # Function to fill the na values using a random element 
    df_missing = df_missing.groupby(grouping_vars).transform(f) # Applying the function to the grouping
    
    return df_missing.merge(df_auxiliar, on='SessionId')


print(len(samp_df))

samp_df = replace_missing(samp_df)
samp_df = replace_missing(samp_df, grouping_vars = ['MunicipalityOrigin', 'edu', 'PopSocio'])
samp_df = replace_missing(samp_df, grouping_vars = ['MunicipalityOrigin', 'PopSocio'])
samp_df = replace_missing(samp_df, grouping_vars = ['PopSocio'])
samp_df[numerical] = samp_df[numerical].fillna(samp_df[numerical].mean())
samp_df[numerical] = samp_df[numerical].astype('int32')

# Drop AgeGroup since we don't need it anymore
samp_df.drop(['AgeGroup'], inplace=True, axis=1)

samp_df = samp_df.dropna(thresh=0.2*len(samp_df), axis=1) # Drop columns with more than 20% of missing values

### Categorical variables define 
categorical = [col for col in list(samp_df) if col not in numerical]

samp_df[categorical] = samp_df[categorical].astype('category')
# Convert categorical to categorical data type
for cat in categorical:
    samp_df[cat] = samp_df[cat].cat.rename_categories({'nan': 'MISS'})
samp_df[categorical] = samp_df[categorical].astype('str')
samp_df[categorical].replace(to_replace=['NaN', 'nan'], value='MISS') # Replace NaN with MISS as a new category on categorical data so we don't have a problem in keras.
samp_df[categorical] = samp_df[categorical].astype('category')

# Second dataset
samp_df.to_csv('sampling_df_no_nan.txt', sep=',', index=False)