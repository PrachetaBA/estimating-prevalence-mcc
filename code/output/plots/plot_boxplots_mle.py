#!/usr/bin/env python3

"""
Created on Mon Mar  22 13:10:20 2021
@author: prachetaamaranath

Title: Plot boxplots for comparing MaxEnt vs. MLE 
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns 

# Parameters for plots
params = {
	'xtick.labelsize': 10,
	'ytick.labelsize': 10,
	'font.size': 8,
	'figure.autolayout': True,
	'figure.figsize': [3.5,3.25],
	'axes.titlesize' : 10,
	'axes.labelsize' : 10,
	'lines.linewidth' : 2,
	'lines.markersize' : 6,
	'legend.fontsize': 13,
	'mathtext.fontset': 'stix',
	'font.family': 'STIXGeneral',
	'text.usetex': True
}
rcParams.update(params)

def plot_mle(): 
	"""
	Plot the figure for MaxEnt vs. MLE 
	"""
	# Create figure

	# Use grayscale images for paper 
	plt.style.use('grayscale')

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
	# df = pd.read_csv('../../data/parameters/power_divergence_3.csv')
	df = pd.read_csv('../../../data/parameters/js_distance_3.csv')

	# Columns of data to plot 
	maxent = df['max_d']
	mle = df['emp_d']

	# Step 3: Plot the boxplots
	bp = ax.boxplot([maxent, mle], showfliers=True, whis='range', notch=False)
	xtick_labels = [r'$\hat{p}$', r'$\hat{p}_{ML}$']
	ax.set_xticklabels(xtick_labels)
	ax.set_ylabel("JS Distance from ground truth")

	plt.savefig('../../figures/mle_plot.pdf',dpi=300)
	# plt.show()

if __name__ == '__main__':
	plot_mle()