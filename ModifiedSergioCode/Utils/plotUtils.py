import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_marginals(real_and_sampled, margin_var, results_dir=None):
	#results_dir = '/home/shgm/JobPopSynth/result_graphs/'
	temp_df = real_and_sampled.groupby(margin_var).sum() # Temporary dataframe to save partial results and not compute everything again 

	# Extract the values for sampled and real distributions, extract the labels for the variable 
	generated = temp_df['count_sampled']
	real      = temp_df['count_real']
	indices = range(len(temp_df))
	names = temp_df.reset_index()[margin_var].cat.categories
	# Calculate optimal width
	width = np.min(np.diff(indices))/3.

	fig = plt.figure(figsize=(25,13))
	ax = fig.add_subplot(111)
	ax.bar(indices-width/2., generated, width,color='b', label='Generated')
	ax.bar(indices+width/2., real, width, color='r', label='Real')
	ax.axes.set_xticklabels(names)
	ax.set_xlabel(margin_var)
	ax.legend()
	if results_dir is not None:
		plt.savefig(results_dir+'margins_{}.png'.format(margin_var))
	plt.show()

def compute_stat(Y_test, Y_pred, do_plot, plot_log, plot_name=None):
	results_dir = '/home/shgm/JobPopSynth/result_graphs/'
	Y_test, Y_pred = np.array(Y_test), np.array(Y_pred)
	corr_mat = np.corrcoef(Y_test, Y_pred)
	corr = corr_mat[0, 1]
	if np.isnan(corr): corr = 0.0
	# MAE
	mae = np.absolute(Y_test - Y_pred).mean()
	# RMSE
	rmse = np.linalg.norm(Y_test - Y_pred) / np.sqrt(len(Y_test))
	# SRMSE
	ybar = Y_test.mean()
	srmse = rmse / ybar
	# r-square
	u = np.sum((Y_pred - Y_test)**2)
	v = np.sum((Y_test - ybar)**2)
	r2 = 1.0 - u / v
	stat = {'mae': mae, 'rmse': rmse, 'r2': r2, 'srmse': srmse, 'corr': corr}
	if do_plot:
		fig = plt.figure(figsize=(3, 3), dpi=200, facecolor='w', edgecolor='k')
		#plot
		print('corr = %f' % (corr))
		print('MAE = %f' % (mae))
		print('RMSE = %f' % (rmse))
		print('SRMSE = %f' % (srmse))
		print('r2 = %f' % (r2))
		min_Y = min([min(Y_test),min(Y_pred)])
		max_Y = max([max(Y_test),max(Y_pred)])
		w = max_Y - min_Y
		max_Y += w * 0.05
		text = ['SRMSE = {:.3f}'.format(stat['srmse']),
		        'Corr = {:.3f}'.format(stat['corr']),
		        '$R^2$ = {:.3f}'.format(stat['r2'])]
		text = '\n'.join(text)
		plt.text(w * 0.08, w * 0.8, text)
		plt.plot(Y_test, Y_pred, '.', alpha=0.5, ms=10, color='seagreen', markeredgewidth=0)
		plt.plot([min_Y, max_Y], [min_Y, max_Y], ls='--', color='gray', linewidth=1.0)
		plt.axis([min_Y, max_Y, min_Y, max_Y])
		plt.xlabel('true')
		plt.ylabel('predicted')
		if plot_log:
		    eps = 1e-6
		    plt.axis([max(min_Y, eps), max_Y, max(min_Y, eps), max_Y])
		    plt.yscale('log')
		    plt.xscale('log')
		if plot_name is not None:
		    plt.savefig(results_dir+'joint_{}.png'.format(plot_name))
		plt.show()
	return stat



