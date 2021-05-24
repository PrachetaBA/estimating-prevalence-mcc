#!/usr/bin/env python3

"""
Created on Thu Feb  25 22:20:20 2021
Modified on Sat Mar 13 13:55:20 2021
@author: prachetaamaranath

Title: Extract power divergence from experiments containing different support parameters or regularization parameters 

Input: 
Keyword signifying data generation process
"support_1": Extract power divergence from unregularized maxent experiments with different support values 
"width": Extract power divergence from regularized maxent experiments with predicted support_1 values 
"support_2": Extract power divergence from regularized maxent experiments with different support values

For each keyword, the functions use different input location folders 
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
def read_maxent_prob_support(filename):
	try:
		with open(filename, "rb") as outfile:
			data = pickle.load(outfile)				# If fails, use encoding='latin1'
		return (data[0], data[3], data[4])
	except IOError as e:
		return 'NaN'
	except EOFError as e:
		return 'NaN'

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob_width(filename):
	try:
		with open(filename, "rb") as outfile:
			data = pickle.load(outfile)				# If fails, use encoding='latin1'
		return data[0]
	except IOError as e:
		return 'NaN'
	except EOFError as e:
		return 'NaN'

# Read the maxent prob. distribution for sum of diseases as well as empirical probability distribution
def read_maxent_prob_mle(filename):
	try:
		with open(filename, "rb") as outfile:
			data = pickle.load(outfile)				# If fails, use encoding='latin1'
		return data[0]
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

def calc_div_mle(f): 
	"""
	INPUT:
	------
	f (int): File number ranging from 1-2100

	OUTPUT:
	-------
	Returns the power divergence of maxent vs. true probability distribution 
	as well as MLE vs. true probability distribution, returns NaN if there is an error
	"""
	maxent_file = 'output/mle_expts/synthetic_data_'+str(f)+'.pickle'

	# Check if directory exists 
    expts_directory = 'output/mle_expts/'
    if not os.path.isdir(expts_directory): 
        os.makedirs(expts_directory)
        print("Create directory: ", expts_directory)

	output = read_maxent_prob_mle(maxent_file)

	# Sanity Check: Errors in the file 
	if output == 'NaN':
		return 'NaN'
	else: 
		maxent_dist = output 

	emp_file = 'output/mle_expts/synthetic_data_'+str(f)+'_mle.pickle'
	output = read_maxent_prob_mle(emp_file)

	# Sanity Check: Errors in the file 
	if output == 'NaN':
		return 'NaN'
	else: 
		emp_dist = output 


	# Step 2: Retrieve true probability distribution
	true_file = '../data/true_data_3/truedist_expt_'+str(f)+'.pickle'
	true_dist = read_true_prob(true_file)

	# Step 3: Calculate power divergence 
	p = np.array(maxent_dist)
	q = np.array(emp_dist)
	r = np.array(true_dist)
	# max_pd, _ = power_divergence(f_obs=p, f_exp=r, lambda_="cressie-read")
	# emp_pd, _ = power_divergence(f_obs=q, f_exp=r, lambda_='cressie-read')

	max_pd = distance.jensenshannon(p, r)
	emp_pd = distance.jensenshannon(q, r)

	# Return power divergence for both maxent and for MLE
	return (max_pd, emp_pd)


def calc_div_support(f, s, keyword):
	"""	
	INPUT:
	------
	f (int): File number ranging from 1-4200
	s (int): Support number ranging from 1-10
	keyword (str): Keyword specifiying which input location folder to access

	OUTPUT:
	-------
	Returns the power divergence of maxent vs. true probability distribution, returns NaN if 
	there is an error
	"""
	
	# Sanity check: if directory exists, else create it 
    expts1_directory = 'output/support_expts_1/'
    if not os.path.isdir(expts1_directory): 
        os.makedirs(expts1_directory)
        print("Create directory: ", expts1_directory)

    expts2_directory = 'output/support_expts_2/'
    if not os.path.isdir(expts2_directory): 
        os.makedirs(expts2_directory)
        print("Create directory: ", expts2_directory)

	# Step 1: Retrieve maxent probability distribution
	if keyword == 'support_1':
		maxent_file = 'output/support_expts_1/synthetic_data_'+str(f)+'_s'+str(s)+'.pickle'
	elif keyword == 'support_2':
		maxent_file = 'output/support_expts_2/synthetic_data_'+str(f)+'_s'+str(s)+'.pickle'

	output = read_maxent_prob_support(maxent_file)

	# Sanity Check: Errors in the file 
	if output == 'NaN':
		return 'NaN'
	else: 
		maxent_dist, constraints, support = output 

	# Step 2: Retrieve true probability distribution
	if keyword == 'support_1':
		true_file = '../data/true_data_1/truedist_expt_'+str(f)+'.pickle'
	elif keyword == 'support_2':
		true_file = '../data/true_data_2/truedist_expt_'+str(f)+'.pickle'
	true_dist = read_true_prob(true_file)

	if true_dist == 'NaN':
		return 'NaN'

	# Step 3: Calculate power divergence 
	p = np.array(maxent_dist)
	q = np.array(true_dist)
	# pow_div, _ = power_divergence(f_obs=p, f_exp=q, lambda_="cressie-read")
	js_div = distance.jensenshannon(p, q)

	# Return power divergence, support and constraints 
	# return (pow_div, support, constraints)
	return (js_div, support, constraints)

def calc_div_width(f, w):
	"""	
	INPUT:
	------
	f (int): File number ranging from 1-4200
	w (int): Width number ranging from 0-10

	OUTPUT:
	-------
	Returns the power divergence of maxent vs. true probability distribution, returns NaN if 
	there is an error
	"""
	
	# Sanity check if directory exists 
    reg_directory = 'output/reg_expts/'
    if not os.path.isdir(reg_directory): 
        os.makedirs(reg_directory)
        print("Create directory: ", reg_directory)

	# Step 1: Retrieve maxent probability distribution
	if w == 0:
		maxent_file = 'output/reg_expts/synthetic_data_'+str(f)+'_ur.pickle'
	else: 
		maxent_file = 'output/reg_expts/synthetic_data_'+str(f)+'_w'+str(w)+'.pickle'
	output = read_maxent_prob_width(maxent_file)

	# Sanity Check: Errors in the file 
	if output == 'NaN':
		return 'NaN'
	else: 
		maxent_dist = output 

	# Step 2: Retrieve true probability distribution
	true_file = '../data/true_data_1/truedist_expt_'+str(f)+'.pickle'
	true_dist = read_true_prob(true_file)

	if true_dist == 'NaN':
		return 'NaN'

	# Step 3: Calculate power divergence 
	p = np.array(maxent_dist)
	q = np.array(true_dist)
	# pow_div, _ = power_divergence(f_obs=p, f_exp=q, lambda_="cressie-read")
	js_div = distance.jensenshannon(p, q)

	# Return power divergence, support and constraints 
	# return pow_div
	return js_div


def main(keyword): 
	"""
	OUTPUT:
	-------
	Create a new training data file that contains all features
	f_n, s_n, d, N, pi_s, pow_div, s, c
	or 
	f_n, w_n, d, N, pi_s, pow_div
	"""

	if keyword == 'width': 
		# Parameters file 
		par_file = '../data/parameters/parameters_1.csv'
		# Width Parameters file 
		width_file = '../data/parameters/width_parameters_1.csv'

		# Step 1: Read parameters from power divergence 
		df = pd.read_csv(par_file)

		# Step 2: Write into a new power divergence file 
		with open(width_file, "w") as csvFile: 
			first_row = ['f_n', 'w', 'd', 'N', 'pi_s', 'div']
			csv.writer(csvFile).writerow(first_row)

		# Step 3: For every combination of f and s as present in the parameters subset file, append the power divergence 
		for idx, row in df.iterrows(): 
			f_n = row[0]
			d = row[1]
			N = row[2]
			pi_s = row[3]
			for w in range(16):
				op = calc_div_width(int(f_n), w)

				if op != 'NaN': 
					# Write parameters to the file 
					row = [f_n, w, d, N, pi_s, op]
					with open(width_file, "a") as csvFile: 
						csv.writer(csvFile).writerow(row)
				else: 
					row = [f_n, w, d, N, pi_s, -1]
					with open(width_file, "a") as csvFile: 
						csv.writer(csvFile).writerow(row)
		csvFile.close()


	elif keyword == "support_1" or keyword == "support_2":
		if keyword == 'support_1':
			# Support parameters file 
			supp_file = '../data/parameters/support_parameters_1.csv'
			# Power divergence file 
			# pow_file = '../data/parameters/power_divergence_1.csv'
			pow_file = '../data/parameters/js_distance_1.csv'
		elif keyword == 'support_2': 
			# Support parameters file 
			supp_file = '../data/parameters/support_parameters_2.csv'
			# Power divergence file 
			# pow_file = '../data/parameters/power_divergence_2.csv'
			pow_file = '../data/parameters/js_distance_2.csv'

		# Step 1: Read parameters from support file 
		df = pd.read_csv(supp_file)

		# Step 2: Write into new power divergence file 
		with open(pow_file, "w") as csvFile: 
			first_row = ['f_n','s_n','d','N','pi_s','div','s','c']
			csv.writer(csvFile).writerow(first_row)

		# Step 3: For every combination of f and s append the power divergence and constraints
		for idx, row in df.iterrows(): 
			f_n = row[0]
			s_n = row[1]
			d = row[2]
			N = row[3]
			pi_s = row[4]
		
			op = calc_div_support(int(f_n), int(s_n), keyword)

			if op != 'NaN':
				# Write parameters onto file
				p_d, s, c = op
				row = [f_n, s_n, d, N, pi_s, p_d, s, c]
				with open(pow_file, "a") as csvFile: 	
					csv.writer(csvFile).writerow(row)
			else: 
				# Write parameters onto file
				row = [f_n, s_n, d, N, pi_s, -1, -1, -1]
				with open(pow_file, "a") as csvFile: 	
					csv.writer(csvFile).writerow(row)

		# Close the file 
		csvFile.close()

	else: 
		if keyword == 'mle':
			# Parameters file 
			par_file = '../data/parameters/parameters_3.csv'
			# Power divergence file 
			# pow_file = '../data/parameters/power_divergence_3.csv'
			pow_file = '../data/parameters/js_distance_3.csv'

			# Step 1: Read parameters from paramters file 
			df = pd.read_csv(par_file)

			# Step 2: Write into a new power divergence file 
			with open(pow_file, "w") as csvFile:
				first_row = ['f_n','d','N', 'pi_s', 'max_d', 'emp_d']
				csv.writer(csvFile).writerow(first_row)

			# Step 3: For every f, append the power divergence 
			for idx, row in df.iterrows(): 
				f_n = row[0]
				d = row[1]
				N = row[2]
				pi_s = row[3]

				op = calc_div_mle(int(f_n))

				if op != 'NaN': 
					# Write parameters to the file
					max_pd, emp_pd = op 
					row = [f_n, d, N, pi_s, max_pd, emp_pd]
					with open(pow_file, "a") as csvFile: 
						csv.writer(csvFile).writerow(row)
				else: 
					row = [f_n, d, N, pi_s, -1, -1]
					with open(pow_file, "a") as csvFile: 
						csv.writer(csvFile).writerow(row)
			# Close file 
			csvFile.close()

if __name__ == '__main__':
	keyword = sys.argv[1]
	main(keyword)
