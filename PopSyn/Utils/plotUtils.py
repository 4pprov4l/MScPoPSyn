import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# from bokeh.models import ColumnDataSource, HoverTool
# from bokeh.palettes import HighContrast3
# from bokeh.plotting import figure, show
# from bokeh.layouts import column, gridplot
# from bokeh.transform import dodge

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



def plot_column_distribution(column, compare_mode = 'absolute'):

    if compare_mode == 'absolute':
        df = column.value_counts().to_frame().sort_values(by=column.name, ascending=True).reset_index()

        source = ColumnDataSource(df)
        hover = HoverTool(tooltips=[(column.name, f"@{column.name}"), ("# of People", "@count")])
    
        p = figure(x_range=df[column.name].astype(str).tolist(),  title=f"Distribution of {column.name}")
        p.vbar(x=column.name, top='count', width=0.9,source=source, line_color='white', fill_color=HighContrast3[2], sizing_mode='stretch_both')
        p.add_tools(hover)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.y_range.end = df['count'].max() + 100
        p.xaxis.major_label_orientation = 0.5

    if compare_mode == 'percent':

        df = column.value_counts().to_frame().sort_values(by=column.name, ascending=True).reset_index()
        df['percent'] = (df['count']/df['count'].sum())*100

        source = ColumnDataSource(df)
        hover = HoverTool(tooltips=[(column.name, f"@{column.name}"), ("# of People", "@count"),("Percent of People", "@percent")])
    
        p = figure(x_range=df[column.name].astype(str).tolist(),  title=f"Distribution of {column.name}")
        p.vbar(x=column.name, top='percent', width=0.9,source=source, line_color='white', fill_color=HighContrast3[1], sizing_mode='stretch_both')
        p.add_tools(hover)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.y_range.end = df['percent'].max() + 1
        p.xaxis.major_label_orientation = 0.5
    return p



def plot_column_distribution_offset(column1, column2, compare_mode='absolute'):

    df1 = column1.value_counts().to_frame().sort_values(by=column1.name, ascending=True).reset_index()
    df1['percent'] = (df1['count']/df1['count'].sum())*100
    df2 = column2.value_counts().to_frame().sort_values(by=column2.name, ascending=True).reset_index()
    df2['percent'] = (df2['count']/df2['count'].sum())*100


    if set(df1[column1.name].unique()) == set(df2[column2.name].unique()):
        print("The unique values of the two columns are the same.")
        # df = pd.merge(df1, df2, on=column1.name)
        source1 = ColumnDataSource(df1)
        source2 = ColumnDataSource(df2)

    else:
        print("The unique values of the two columns are different.")
        return


    if compare_mode == 'absolute':

        p = figure(x_range=df1[column1.name].astype(str).tolist(), title=f"Distribution of {column1.name} and {column2.name}")
        p.vbar(x=dodge(column1.name, -0.15, range=p.x_range), top='count', source=source1, width=0.2, color="#c9d9d3", legend_label=f"Real data column: {column1.name}", sizing_mode='stretch_both')
        p.vbar(x=dodge(column2.name, 0.15, range=p.x_range), top='count', source=source2, width=0.2, color="#718dbf", legend_label=f"WGAN data column: {column2.name}", sizing_mode='stretch_both')


        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        p.xaxis.major_label_orientation = 0.5
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'


    if compare_mode == 'percent':

        p = figure(x_range=df1[column1.name].astype(str).tolist(), title=f"Distribution of {column1.name} and {column2.name}")
        p.vbar(x=dodge(column1.name, -0.15, range=p.x_range), top='percent', source=source1, width=0.2, color="#c9d9d3", legend_label=f"Real data column: {column1.name}", sizing_mode='stretch_both')
        p.vbar(x=dodge(column2.name, 0.15, range=p.x_range), top='percent', source=source2, width=0.2, color="#718dbf", legend_label=f"WGAN data column: {column2.name}", sizing_mode='stretch_both')

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        p.xaxis.major_label_orientation = 0.5
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'

    return p





def plot_distribution_comparison(column1, column2, compare_mode = 'absolut', style = 's2s'):
    
    if style == 's2s':

        # Plot for column1
        p1 = plot_column_distribution(column1, compare_mode = compare_mode)
    
        # Plot for column2
        p2 = plot_column_distribution(column2, compare_mode = compare_mode)

        # Show the plots side by side in a grid
        plot = gridplot([p1, p2], ncols=2, width=250, height=250)

    if style == 'offset':

        # Plot for column1
        plot = plot_column_distribution_offset(column1, column2, compare_mode = compare_mode)


    return plot