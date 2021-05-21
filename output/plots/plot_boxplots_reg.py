#!/usr/bin/env python3

"""
Created on Mon Mar  22 13:10:20 2021
@author: prachetaamaranath

Title: Create publication-style boxplots for different regularization parameters for d=9
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns 

# Parameters for plots
params = {
   'axes.labelsize': 12,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [10, 7]
}
rcParams.update(params)

def plot_d9(): 
	"""
	Plot the figure for d = 9 
	"""
	# Create figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# More settings
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.tick_params(axis='x', direction='out')
	ax.tick_params(axis='y', length=0)
	ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
	ax.set_axisbelow(True)

	# Step 1: Read the power divergence parameters from the width_parameters file 
	df = pd.read_csv('../../data/parameters/width_parameters_1.csv')	

	# Step 2: Filter all extreme outliers (>0 and <70,000) specifically for d = 9 
	filter = (df['div'] >= 0.0) & (df['d'] == 9)
	df = df.loc[filter]

	# Columns of data to plot 
	w0 = df.loc[df['w'] == 0]['div']
	w1 = df.loc[df['w'] == 1]['div']
	w2 = df.loc[df['w'] == 2]['div']
	w3 = df.loc[df['w'] == 3]['div']
	w4 = df.loc[df['w'] == 4]['div']
	w5 = df.loc[df['w'] == 5]['div']
	w6 = df.loc[df['w'] == 6]['div']
	w7 = df.loc[df['w'] == 7]['div']
	w8 = df.loc[df['w'] == 8]['div']
	w9 = df.loc[df['w'] == 9]['div']
	w10 = df.loc[df['w'] == 10]['div']
	w11 = df.loc[df['w'] == 11]['div']
	w12 = df.loc[df['w'] == 12]['div']
	w13 = df.loc[df['w'] == 13]['div']
	w14 = df.loc[df['w'] == 14]['div']
	w15 = df.loc[df['w'] == 15]['div']

	# Step 3: Plot the boxplots
	bp = ax.boxplot([w0,w11, w1, w2, w3], showfliers=True)

	xtick_labels = ['Unreg','10', '1', '1e-1', '1e-2']
	ax.set_xticklabels(xtick_labels)
	ax.set_xlabel("W")
	ax.set_ylabel("JS distance from ground truth")
	ax.set_title("d = 9, With outliers")

	plt.style.use('grayscale')
	# plt.savefig('W-boxplot-d9.pdf',dpi=300)
	plt.show()

def plot_all(): 
	# Read CSV file 
	# Step 1: Read the power divergence parameters from the width_parameters file 
	df = pd.read_csv('../../data/parameters/width_parameters_1.csv')

	# Step 2: Filter all extreme outliers (>0) 
	filter = (df['div'] >= 0.0)
	df = df.loc[filter]

	# Columns of data to plot 
	w0 = df.loc[df['w'] == 0]['div']
	w1 = df.loc[df['w'] == 1]['div']
	w2 = df.loc[df['w'] == 2]['div']
	w3 = df.loc[df['w'] == 3]['div']
	w4 = df.loc[df['w'] == 4]['div']
	w5 = df.loc[df['w'] == 5]['div']
	w6 = df.loc[df['w'] == 6]['div']
	w7 = df.loc[df['w'] == 7]['div']
	w8 = df.loc[df['w'] == 8]['div']
	w9 = df.loc[df['w'] == 9]['div']
	w10 = df.loc[df['w'] == 10]['div']
	w11 = df.loc[df['w'] == 11]['div']
	w12 = df.loc[df['w'] == 12]['div']
	w13 = df.loc[df['w'] == 13]['div']
	w14 = df.loc[df['w'] == 14]['div']
	w15 = df.loc[df['w'] == 15]['div']

	# Create figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# More settings
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.tick_params(axis='x', direction='out')
	ax.tick_params(axis='y', length=0)
	ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
	ax.set_axisbelow(True)
	bp = ax.boxplot([w0, w15, w14, w13, w12, w11, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10], showfliers=False)

	xtick_labels = ['Unreg','1e5', '1e4', '1e3', '1e2', '10', '1', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6','1e-7','1e-8','1e-9']
	ax.set_xticks(np.arange(1, 17))
	ax.set_xticklabels(xtick_labels)
	ax.set_xlabel("W")
	ax.set_ylabel("JS distance from ground truth")
	ax.set_title("All values of d, W")
	plt.style.use('grayscale')
	# plt.savefig('figures/W-boxplot-all-d.pdf',dpi=300)
	plt.show()


def plot_all_split():
	"""
	Plot figure split by diseases 
	"""
	# Create figure 
	fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))
	
	def plot_ind(r,c, d):
		ax[r,c].spines['top'].set_visible(False)
		ax[r,c].spines['right'].set_visible(False)
		ax[r,c].spines['left'].set_visible(True)
		ax[r,c].get_xaxis().tick_bottom()
		ax[r,c].get_yaxis().tick_left()
		ax[r,c].tick_params(axis='x', direction='out')
		ax[r,c].tick_params(axis='y', length=0)
		ax[r,c].grid(axis='y', color="0.9", linestyle='-', linewidth=1)
		ax[r,c].set_axisbelow(True)

		# Step 1: Read the power divergence parameters from the width_parameters file 
		df = pd.read_csv('../../data/parameters/width_parameters_1.csv')

		# Step 2: Filter all extreme outliers (>0 and <70,000) specifically for d = 9 
		filter = (df['div'] >= 0.0) & (df['d'] == d)
		df = df.loc[filter]

		# Columns of data to plot 
		w0 = df.loc[df['w'] == 0]['div']
		w1 = df.loc[df['w'] == 1]['div']
		w2 = df.loc[df['w'] == 2]['div']
		w3 = df.loc[df['w'] == 3]['div']
		w4 = df.loc[df['w'] == 4]['div']
		w5 = df.loc[df['w'] == 5]['div']
		w6 = df.loc[df['w'] == 6]['div']
		w7 = df.loc[df['w'] == 7]['div']
		w8 = df.loc[df['w'] == 8]['div']
		w9 = df.loc[df['w'] == 9]['div']
		w10 = df.loc[df['w'] == 10]['div']
		w11 = df.loc[df['w'] == 11]['div']
		w12 = df.loc[df['w'] == 12]['div']
		w13 = df.loc[df['w'] == 13]['div']
		w14 = df.loc[df['w'] == 14]['div']
		w15 = df.loc[df['w'] == 15]['div']

		# Step 3: Plot the boxplots
		# bp = ax[r,c].boxplot([w15, w14, w13, w12, w11, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10], showfliers=False)
		# ax[r,c].set_xticks(np.arange(1, 17))
		# xtick_labels = ['1e5', '1e4', '1e3', '1e2', '10','0', '1', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6','1e-7','1e-8','1e-9']
		
		bp = ax[r,c].boxplot([w0, w11, w1, w2, w3, w4, w5], showfliers=False)
		ax[r,c].set_xticks(np.arange(1, 8))
		xtick_labels = ['Un-reg', '10', '1', '1e-1', '1e-2', '1e-3', '1e-4']
		
		ax[r,c].set_xticklabels(xtick_labels)
		ax[r,c].set_title("d = "+str(d))

	plot_ind(0, 0, 9)
	plot_ind(0, 1, 10)
	plot_ind(0, 2, 11)
	plot_ind(1, 0, 12)
	plot_ind(1, 1, 13)
	plot_ind(1, 2, 14)
	plot_ind(2, 0, 15)
	plot_ind(2, 1, 16)
	plot_ind(2, 2, 17)
	plot_ind(3, 0, 18)
	plot_ind(3, 1, 19)
	plot_ind(3, 2, 20)
	
	fig.text(0.5, 0.05, 'W', ha='center', size=12)
	fig.text(0.06, 0.5, 'Jensen-shannon distance from ground truth', va='center', rotation='vertical', size=12)
	# plt.savefig('figures/W-boxplot-all-d-split.pdf', dpi=300)
	plt.show()

if __name__ == '__main__':
	plot_d9()
	plot_all()