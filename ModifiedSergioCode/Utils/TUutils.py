import numpy as np
import pandas as pd
from itertools import groupby

# This script contains functions that are useful for the pre-processing of the TU data.


def numerical_binning(df, cols, q=5):
	r"""
	Bins the numerical variables into q categories using quantile cuts

	Arguments: 
		df: The dataset to be binned
		cols: The columns of df to be binned
		q: The number of cuts
	"""
	for var in cols:
		try:
		    pd.qcut(df[var], q=2)
		    df[var] = pd.cut(df[var], bins=np.quantile(df[var], q=np.linspace(0, 1, num=q)), duplicates='drop', include_lowest=True) # If you want to force it to be of q categories, delete duplicates='drop'
		except:
		    df.drop([var], axis=1, inplace=True)
		    print(var, 'dropped')
		
	return df


def encode_onehot(df, cols=None):
	"""
	Returns the one-hot encoded dataset

	Arguments:
		df: The dataset to be one-hot-encoded
		cols: The variables of df to be one-hot encoded
	"""
	if cols is None:
		cols=list(df)
	df = pd.get_dummies(data=df, columns=cols, sparse=True) ### The sparse=True option might be taken off when using the server
	return df


def back_from_dummies(df):
	r"""
	Returns from one-hot encoding into the original dataset

	Arguments:
		df: the dataset to be returned into a normal one
	"""
	result_series = {}

	# Find dummy columns and build pairs (category, category_value)
	dummmy_tuples = [(col.split("_")[0],col) for col in df.columns if "_" in col]

	# Find non-dummy columns that do not have a _
	non_dummy_cols = [col for col in df.columns if "_" not in col]

	# For each category column group use idxmax to find the value.
	for dummy, cols in groupby(dummmy_tuples, lambda item: item[0]):

		#Select columns for each category
		dummy_df = df[[col[1] for col in cols]]

		# Find max value among columns
		max_columns = dummy_df.idxmax(axis=1)

		# Remove category_ prefix
		result_series[dummy] = max_columns.apply(lambda item: item.split("_")[1])

	# Copy non-dummy columns over.
	for col in non_dummy_cols:
		result_series[col] = df[col]

	# Return dataframe of the resulting series
	return pd.DataFrame(result_series)


def data_creator(df, numerical, train_prop=0.5, val_prop=0.5, binning=True, condition_on=None, quantiles=15):
	r"""
	Function that takes a dataset, bins or standarizes its numerical variables, and splits into train validation and test sets

	Arguments: 
	df: the dataset to be processed
	numerical: list of columns which contain numerical values
	train_prop: the proportion of the dataset to be used as training set
	val_prop: the proportion of the training dataset to be used as validation set
	binning: whether to bin or not the numerical variables
	condition_on: name of a variable to be left at the end of the dataset, this is important for conditional VAE or GAN purposes
	"""
	total_df = df.copy()

	# Leaving the conditioned variable at the end of the dataset so we have control over it
	if condition_on is not None:
		cols = list(total_df.columns.values) #Make a list of all of the columns in the df
		cols.pop(cols.index(condition_on))   #Remove conditioning variable from the list
		total_df    = total_df[cols+[condition_on]] # Create a new dataframe with the conditioning column at the end
		cond_labels = total_df[condition_on].cat.categories # Categories of variable of interest

	# Binning or scaling numerical variables
	if numerical is not None:
		if binning:
		    pre_one_hot_df = numerical_binning(total_df, numerical, q=quantiles)
		else:
		    pre_one_hot_df = total_df
		    pre_one_hot_df[numerical] = scaler.fit_transform(total_df[numerical])
	else: 
		pre_one_hot_df = total_df
	one_hot_df = encode_onehot(pre_one_hot_df) # One hot encoding variables 
	col_names  = one_hot_df.columns  

	validation, train, test = np.split(one_hot_df.sample(frac=1).values, [int(val_prop*train_prop*len(df)), int(train_prop*len(df))]) # Splitting into train, validation and test 

	print('Train shape is:', train.shape)
	print('Validation shape is:', validation.shape)
	print('Test shape is:', test.shape)
	return train, test, validation, pre_one_hot_df, one_hot_df, col_names


def samples_to_df(samples, col_names, original_df, pre_one_hot_df, print_duplicates=False, binning=True, numerical=None):
	r"""
	Converts an array of samples back into the original dataset format

	Arguments:
		samples: the samples to be turned into the dataframe
		col_names: the names of the variables from the one-hot-encoded dataset
		original_df : The dataset with the format we want to retrieve
		pre_one_hot_df: The dataset before one-hot encoding  
		print_duplicates (boolean): whether the frunction prints or not duplicate values on the dataset
		binning (boolean): whether the numerical variables from the dataset where transformed using the numerical_binning function
		numerical: if binning is False and there are numerical variables in the dataset, their names so they can be destandarized
	"""
	total_samp_df = pd.DataFrame(data=samples.transpose(), columns=col_names) # Converting array into df, uses the column names AFTER one hot encoding
	total_samp_df = back_from_dummies(total_samp_df) # Going back from dummies to single categorical variables
	if not binning:
		total_samp_df[numerical] = scaler.inverse_transform(total_samp_df[numerical]) # Destandarizing numerical variables
	object_cols = total_samp_df.columns[total_samp_df.dtypes=='object'] # Defining categorical variables
	total_samp_df[object_cols]= total_samp_df[object_cols].astype('category') # Converting their dtype

	for var in object_cols:
		if binning:
		    original_cats = [str(cat) for cat in list(pre_one_hot_df[var].cat.categories)]
		else:
		    original_cats = [str(cat) for cat in list(original_df[var].cat.categories)]
		total_samp_df[var] = total_samp_df[var].cat.add_categories([cat for cat in original_cats if cat not in total_samp_df[var].cat.categories])

	if print_duplicates:
		print('Number of duplicate samples:', sum(total_samp_df.duplicated()))
		
	return total_samp_df


