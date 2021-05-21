#!/usr/bin/env python3

"""
Created on Thu Feb  25 22:20:20 2021
@author: prachetaamaranath
"""

import numpy as np
import pickle, csv, sys, os.path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import power_divergence
from scipy.spatial import distance

# Raise error if power divergence calculation fails 
np.seterr(all='raise')

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob(filename):
	try:
		with open(filename, "rb") as outfile:
			data = pickle.load(outfile)				# If fails, use encoding='latin1'
		return (data[0], data[3], data[4])
	except IOError as e:
		return 'NaN'
	except EOFError as e:
		return 'NaN'

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	try:
		with open(filename, "rb") as outfile:
			prob = pickle.load(outfile) 			# If fails, use encoding='latin1'
		return prob[0]
	except IOError as e:
		return 'NaN'
	except EOFError as e:
		return 'NaN'

def calc_div(f, s):
	"""	
	INPUT:
	------
	f (int): File number ranging from 1-4200
	s (int): Support number ranging from 1-10

	OUTPUT:
	-------
	Returns the power divergence of maxent vs. true probability distribution, returns NaN if 
	there is an error
	"""
	
	# Step 1: Retrieve maxent probability distribution
	maxent_file = '../output/support_expts/synthetic_data_'+str(f)+'_s'+str(s)+'.pickle'
	output = read_maxent_prob(maxent_file)

	# Sanity Check: Errors in the file 
	if output == 'NaN':
		return 'NaN'
	else: 
		maxent_dist, constraints, support = output 

	# Step 2: Retrieve true probability distribution
	true_file = '../data/true_data/truedist_expt_'+str(f)+'.pickle'
	true_dist = read_true_prob(true_file)

	if true_dist == 'NaN':
		return 'NaN'

	# Step 3: Calculate power divergence 
	p = np.array(maxent_dist)
	q = np.array(true_dist)
	pow_div, _ = power_divergence(f_obs=p, f_exp=q, lambda_="cressie-read")

	# Return power divergence, support and constraints 
	return (pow_div, support, constraints)


def main(): 
	"""
	OUTPUT:
	-------
	Create a new training data file that contains all features
	f_n, s_n, d, N, pi_s, pow_div, s, c
	"""

	# Support parameters file 
	supp_file = '../data/parameters/support_parameters.csv'

	# Power divergence file 
	pow_file = '../data/parameters/power_divergence.csv'


	# Step 1: Read parameters from support file 
	df = pd.read_csv(supp_file)

	# Step 2: Write into new power divergence file 
	with open(pow_file, "w") as csvFile: 
		first_row = ['f_n','s_n','d','N','pi_s','pow_div','s','c']
		csv.writer(csvFile).writerow(first_row)

	# Step 3: For every combination of f and s append the power divergence and constraints
	f_counter = 1
	for file in range(df.shape[0]): 
		for s_n in range(10):
			d = df.iloc[file]['d']
			N = df.iloc[file]['N']
			pi_s = df.iloc[file]['pi_s']
			
			op = calc_div(f_counter, s_n+1)

			if op != 'NaN':
				# Write parameters onto file
				p_d, s, c = op
				row = [f_counter, s_n+1, d, N, pi_s, p_d, s, c]
				with open(pow_file, "a") as csvFile: 	
					csv.writer(csvFile).writerow(row)
			else: 
				# Write parameters onto file
				row = [f_counter, s_n+1, d, N, pi_s, -1, -1, -1]
				with open(pow_file, "a") as csvFile: 	
					csv.writer(csvFile).writerow(row)

		# Increment file counter
		f_counter += 1

	# Close the file 
	csvFile.close()

if __name__ == '__main__':
	main()