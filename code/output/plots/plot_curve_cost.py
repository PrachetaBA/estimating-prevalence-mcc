# Import libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns 
import os 
from scipy.optimize import curve_fit
import scipy.stats as stats
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
"""

import matplotlib as mpl
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
""" 

def exponential_fit(): 
	# Exponential Fit for MEPS data 
	# Parameters for plots
	params1 = {
		'xtick.labelsize': 20,
		'ytick.labelsize': 20,
		'font.size': 22,
		'figure.autolayout': True,
		'figure.figsize': [10.5,6.25],
		'axes.titlesize' : 22,
		'axes.labelsize' : 22,
		'lines.linewidth' : 2,
		'lines.markersize' : 6,
		'legend.fontsize': 22,
		'mathtext.fontset': 'stix',
		'font.family': 'STIXGeneral',
		'text.usetex': True
	}

	# SMALL_SIZE = 16
	# MEDIUM_SIZE = 20
	# BIGGER_SIZE = 22

	# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

	rcParams.update(params1)
	plt.style.use('grayscale')

	# Function to extract data from csv file
	def get_data(data):
		tups = [tuple(x) for x in data.values]
		data_arr = np.asarray(tups)
		data_arr = data_arr.astype(int)

		# Map all positive values to 1 since any > 0 indicates the disease
		data_arr[data_arr > 0] = 1
		return data_arr

	# Function to return marginals for every disease 
	def curve_stats(data_arr):
		num_dis = data_arr.shape[1]
		sum_diseases = np.zeros(num_dis + 1)
		for vec in data_arr:
			j = sum(vec)
			sum_diseases[j] += 1
			
		sum_diseases /= data_arr.shape[0]
		return sum_diseases

	# Perform curve_fit using scipy.optimize curve fitting function and plot data
	def curve_fitting_func(data_arr):
		N = data_arr.shape[1]
		x = np.arange(0, data_arr.shape[1]+1)
		y = curve_stats(data_arr)
		
		# a = loc parameter, b= scale parameter
		def func(x, a,b):
			return a*np.exp(-b*x)

		popt, pcov = curve_fit(func, x, y)
	 
		# adding the artificially generated data 
		print("loc is : ", popt[0])
		print("scale is: ", popt[1])
		alpha = []
		for i in range(1, N+2):
			alpha.append(stats.expon.pdf(i, loc=popt[0], scale=1/popt[1]))
		ans = np.array(alpha)/sum(alpha)
		
		# Plotting the curve-fit
		plt.close('all')
		fig, ax = plt.subplots()
		
		alphaVal = 1.0
		linethick = 2.0
		
		ax.plot(x, y, 'o',
	#             color=CB_color_cycle[0], 
				label="MEPS",
				lw=2.0,
				alpha=alphaVal)
		
		ax.plot(x, func(x, *popt), 
	#             color=CB_color_cycle[1], 
				linestyle = '-',
				lw=linethick,
				label="Exponential fitted curve",
				alpha=alphaVal)
		
		ax.set_xlabel('disease cardinality')
		ax.set_ylabel('Proportion')
	#     ax.set_title('MEPS - Marginal Probabilities Curve Fit', fontdict={'fontsize':20})
		ax.set_xticks(np.arange(0,21,2))
		ax.yaxis.set_major_formatter(ScalarFormatter())
		ax.yaxis.major.formatter._useMathText = True
		ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False)
	#     plt.savefig('MEPS-exp.png',dpi=300)
		plt.savefig('../../figures/MEPS-exp.pdf',dpi=300)
		# plt.show()
		return popt[0], popt[1]

	def top_x_dis(data,x):
		counts = np.sum(data, axis=0)
		counts = counts.sort_values(ascending=False)[:x]
		cols = counts.index
		new_data = df[cols]
		return new_data

	# Plotting the prevalence MEPS data 
	df = pd.read_csv('../../../data/meps_data_prevalence.csv')
	d_arr = get_data(df)
	s_dis = curve_stats(d_arr)

	popt_1, popt_2 = curve_fitting_func(d_arr)

def cost_fig(): 
	params1 = {
		'xtick.labelsize': 28,
		'ytick.labelsize': 28,
		'font.size': 28,
		'figure.autolayout': True,
		'figure.figsize': [13,8],
		'axes.titlesize' : 28,
		'axes.labelsize' : 28,
		'lines.linewidth' : 2,
		'lines.markersize' : 6,
		'legend.fontsize': 26,
		'mathtext.fontset': 'stix',
		'font.family': 'STIXGeneral',
		'text.usetex': True
	}
	rcParams.update(params1)
	plt.style.use('grayscale')

	df_costs = pd.read_csv('costs.csv', index_col=0)
	fig, ax = plt.subplots()

	alphaVal = 1.0
	linethick = 2.0

	ax.scatter(df_costs.index, df_costs['emp_cost'], marker='o',
			label="Observed MEPS cost",
			linewidths=2.0,
			alpha=alphaVal)

	ax.plot(df_costs.index, 2092.1*df_costs.index + 506.16, linestyle='dashed',
				label="Estimated cost",
				lw=2.0,
				alpha=alphaVal)

	ax.set_xlabel('disease cardinality')
	ax.set_ylabel('Cost (\$)')
	ax.set_xticks(np.arange(0,21,2))
	ax.yaxis.set_major_formatter(ScalarFormatter())
	ax.yaxis.major.formatter._useMathText = True
	ax.legend(loc=0, frameon=True)
	plt.annotate(r'Estimated cost = \\ \\ 2092.1 $\times$ (disease cardinality) + 506.16',
				 xy = (9.5, 19000), 
				 xytext = (10, 14000), 
				 arrowprops = dict(width=2, headwidth=10, facecolor = 'black'),
				 color = 'k', fontsize = 22)
	plt.tight_layout()
	# plt.savefig('meps-cost.png',dpi=300)
	plt.savefig('../../figures/meps-cost.pdf',dpi=300)
	# plt.show()

if __name__ == '__main__':
	# exponential_fit()
	cost_fig()